"""A ResNet that reads a player's whole jersey number from a crop of their torso, an alternative
to the YOLO digit detector (run_models.py --number-reader resnet).

One backbone with three heads, after Vats et al., "Multi-task learning for jersey number
recognition in Ice Hockey" (2021):
    number  101 classes: 0-99, plus NONE (no readable number)
    tens    11 classes: 0-9, plus EMPTY (a one-digit number, or no number)
    units   11 classes: 0-9, plus EMPTY (no number)
The digit heads let a number that is rare in the training data borrow from its digits, which are
common. decode() scores every number on all three heads.

The model sees a fixed band of the player's box (TORSO_Y, the same band the digit detector's
numbers sit in), resized to INPUT_HW. Training (train_jersey.py) and inference both go through
prepare(), so the two always see identical crops.

Weights live in CV_Models/jersey_model.pt (a checkpoint from train_jersey.py) and, on the Jetson,
CV_Models/jersey_model.engine (built by export_engines.py).
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

NONE = 100   # number head: no readable number
EMPTY = 10   # digit heads: no digit in this position
INPUT_HW = (160, 128)    # torso crops are taller than wide
TORSO_Y = (0.05, 0.75)   # fraction of the player box's height the torso crop covers
MAX_BATCH = 16           # player crops per call; the TensorRT engine is built for up to this many
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet, what the pretrained backbone expects
STD = np.array([0.229, 0.224, 0.225], np.float32)
ARCHS = {"resnet18": torchvision.models.resnet18, "resnet34": torchvision.models.resnet34}


class JerseyNet(nn.Module):
    def __init__(self, arch="resnet34", pretrained=True):
        super().__init__()
        self.backbone = ARCHS[arch](weights="IMAGENET1K_V1" if pretrained else None)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(0.2)
        self.number = nn.Linear(features, NONE + 1)
        self.tens = nn.Linear(features, EMPTY + 1)
        self.units = nn.Linear(features, EMPTY + 1)

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        return self.number(f), self.tens(f), self.units(f)


def encode(number):
    """Class ids for each head: "12" -> (12, 1, 2), "7" -> (7, EMPTY, 7), "" or None -> (NONE, EMPTY, EMPTY)."""
    if not number:
        return NONE, EMPTY, EMPTY
    n = int(number)
    return (n, n // 10, n % 10) if len(number) == 2 else (n, EMPTY, n)


# For each number-head class, the tens and units classes that agree with it.
_CANDIDATES = torch.arange(NONE + 1)
_TENS = torch.where((_CANDIDATES >= 10) & (_CANDIDATES < NONE), _CANDIDATES // 10, EMPTY)
_UNITS = torch.where(_CANDIDATES < NONE, _CANDIDATES % 10, EMPTY)


@torch.no_grad()
def decode(number_logits, tens_logits, units_logits):
    """Pick each crop's number by its summed log-probability over the three heads. Returns one
    (number or None, confidence) per crop; the confidence is the number head's probability."""
    num = number_logits.float().log_softmax(-1)
    tens = tens_logits.float().log_softmax(-1)
    units = units_logits.float().log_softmax(-1)
    t, u = _TENS.to(num.device), _UNITS.to(num.device)
    best = (num + tens[:, t] + units[:, u]).argmax(-1)
    conf = num.exp().gather(1, best[:, None])[:, 0]
    return [(None if b == NONE else str(b), c) for b, c in zip(best.tolist(), conf.tolist())]


def torso_box(box):
    """The torso band of a player box (x1, y1, x2, y2), in the same pixel coordinates."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    return x1, y1 + TORSO_Y[0] * h, x2, y1 + TORSO_Y[1] * h


def torso_crop(img, box):
    """The torso band of a player box cut out of a BGR image (empty if the box is off the image)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = torso_box(box)
    return img[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]


def prepare(crop):
    """A BGR crop resized to the model's input, as RGB uint8 (augmentations apply to this)."""
    crop = cv2.resize(crop, INPUT_HW[::-1], interpolation=cv2.INTER_AREA if crop.shape[0] > INPUT_HW[0]
                      else cv2.INTER_LINEAR)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def normalize(batch):
    """uint8 RGB images (N, H, W, 3) -> the float tensor (N, 3, H, W) the model takes."""
    x = torch.from_numpy(np.ascontiguousarray(batch)).float().div_(255)
    return ((x - torch.from_numpy(MEAN)) / torch.from_numpy(STD)).permute(0, 3, 1, 2).contiguous()


def load_checkpoint(path, map_location="cpu"):
    """A JerseyNet from a train_jersey.py checkpoint, in eval mode."""
    ckpt = torch.load(path, map_location=map_location)
    if tuple(ckpt.get("input_hw", INPUT_HW)) != INPUT_HW:
        raise ValueError(f"{path} was trained at {ckpt['input_hw']}, jersey_net.INPUT_HW is {INPUT_HW}")
    model = JerseyNet(ckpt.get("arch", "resnet34"), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def export_onnx(checkpoint, onnx_path):
    """Export a checkpoint to ONNX with a variable batch size, for building a TensorRT engine."""
    import inspect
    model = load_checkpoint(checkpoint)
    kwargs = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(model, torch.zeros(1, 3, *INPUT_HW), str(onnx_path), input_names=["images"],
                      output_names=["number", "tens", "units"], opset_version=17,
                      dynamic_axes={n: {0: "batch"} for n in ("images", "number", "tens", "units")}, **kwargs)
    return onnx_path


class _TensorRTModel:
    """Runs a jersey_model.engine with torch CUDA tensors as its buffers."""

    def __init__(self, path):
        import tensorrt as trt
        self.trt = trt
        with open(path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Could not load {path}; rebuild it with export_engines.py --models jersey --force")
        self.context = self.engine.create_execution_context()
        self.max_batch = self.engine.get_tensor_profile_shape("images", 0)[2][0]
        self.stream = torch.cuda.Stream()

    def _dtype(self, name):
        return torch.float16 if self.engine.get_tensor_dtype(name) == self.trt.DataType.HALF else torch.float32

    def __call__(self, x):
        x = x.to("cuda", self._dtype("images")).contiguous()
        self.context.set_input_shape("images", tuple(x.shape))
        self.context.set_tensor_address("images", x.data_ptr())
        outputs = []
        for name in ("number", "tens", "units"):
            out = torch.empty(tuple(self.context.get_tensor_shape(name)), dtype=self._dtype(name), device="cuda")
            self.context.set_tensor_address(name, out.data_ptr())
            outputs.append(out)
        self.stream.wait_stream(torch.cuda.current_stream())  # x is ready
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs


class JerseyReader:
    """Reads jersey numbers off player torso crops with a TensorRT engine or a PyTorch checkpoint."""

    def __init__(self, path, device):
        self.path = path
        if path.suffix == ".engine":
            self.model = _TensorRTModel(path)
            self.max_batch = self.model.max_batch
            self.device, self.half = "cuda", False  # the engine has its own precision
        else:
            self.device = device
            self.half = str(device).startswith("cuda") or str(device)[:1].isdigit()
            self.model = load_checkpoint(path).to(device)
            if self.half:
                self.model.half()
            self.max_batch = MAX_BATCH

    @torch.no_grad()
    def read(self, crops):
        """One (number or None, confidence) per BGR torso crop."""
        readings = []
        for start in range(0, len(crops), self.max_batch):
            x = normalize(np.stack([prepare(c) for c in crops[start:start + self.max_batch]]))
            if not isinstance(self.model, _TensorRTModel):
                x = x.to(self.device, torch.float16 if self.half else torch.float32)
            readings += decode(*self.model(x))
        return readings
