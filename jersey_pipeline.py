"""Koshkina & Elder's jersey number pipeline (github.com/mkoshkina/jersey-number-pipeline), run
per player track inside run_models.py (--number-reader pipeline). Per player crop:

    1. Centroid-ReID (CV_Models/unused_models/jersey_Num Models/centroid-reid.ckpt, ResNet-50 trained on Market-1501) embeds the
       crop. Crops far from the rest of the track's crops (another player, an occlusion) are
       dropped, with their Gaussian outlier test: 3 rounds, threshold 3.5 (gaussian_outliers.py).
    2. ViTPose-H (CV_Models/unused_models/jersey_Num Models/vitpose-h.pth, COCO keypoints) finds the shoulders and hips. Without
       all four at confidence >= 0.4 the crop is skipped; otherwise the torso is cut from the
       shoulders to the hips, padded 5 px (helpers.generate_crops).
    3. PARSeq (CV_Models/unused_models/jersey_Num Models/jersey.ckpt, their hockey fine-tune; parseq_jersey.py) reads the digits.
    4. The track's readings are combined as theirs are (helpers.process_jersey_id_predictions with
       useBias): readings under 0.2 confidence count for nothing, two-digit numbers weigh 0.61 and
       one-digit 0.39, and a number needs a summed weight above 1 (parseq_jersey.TrackletVotes).

run_models.py's default reader, parseq_jersey.py, is steps 3 and 4 on a fixed crop placed where
ViTPose puts the torso: on our test clips nearly as accurate, at a tenth of the cost.

Between steps 1 and 2, their legibility classifier (CV_Models/unused_models/jersey_Num Models/legibility_resnet34_hockey_*.pth,
trained with the SAM optimizer, github.com/davda54/sam) drops crops without a readable number, so
ViTPose doesn't run on them. Their pipeline runs offline on whole tracklets; here each crop is
judged against the track's crops so far. ViTPose runs without flip testing, which would double its
cost.

The models are rebuilt here in plain PyTorch, so neither mmpose, the centroids-reid code nor
PyTorch Lightning is needed.
"""

import pickle
import types
from collections import defaultdict
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from parseq_jersey import LEGIBILITY_THRESHOLD, LegibilityClassifier, ParseqReader, TrackletVotes, find_legibility_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Their constants
OUTLIER_THRESHOLD = 3.5    # gaussian_outliers.py
OUTLIER_ROUNDS = 3
MAX_TRACK_FEATURES = 200   # crops per track kept for the outlier test
KEYPOINT_CONF = 0.4        # helpers.CONFIDENCE_THRESHOLD
CROP_PADDING = 5           # helpers.PADDING
COCO_L_SHOULDER, COCO_R_SHOULDER, COCO_L_HIP, COCO_R_HIP = 5, 6, 11, 12


def _to_tensor(imgs_rgb, mean, std, device):
    x = torch.from_numpy(np.stack(imgs_rgb)).float().div_(255)
    return ((x - torch.from_numpy(mean)) / torch.from_numpy(std)).permute(0, 3, 1, 2).contiguous().to(device)


# ---------------------------------------------------------------- Centroid-ReID

def _load_lightning_checkpoint(path):
    """torch.load for a PyTorch Lightning checkpoint without Lightning (or yacs) installed: classes
    from missing packages are unpickled as plain dicts. Only the state_dict is needed."""
    class Stub(dict):
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.update(state)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                return type(name, (Stub,), {})

    module = types.SimpleNamespace(Unpickler=Unpickler, load=lambda f, **kw: Unpickler(f, **kw).load(),
                                   __name__="pickle")
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=module)


class CentroidReID(nn.Module):
    """centroids-reid's CTLModel at inference: ResNet-50 with the last stride 1, average pooled,
    then a batch norm (what their centroid_reid.py takes as the feature)."""

    def __init__(self):
        super().__init__()
        base = torchvision.models.resnet50(weights=None)
        base.layer4[0].conv2.stride = (1, 1)
        base.layer4[0].downsample[0].stride = (1, 1)
        base.fc = nn.Identity()
        self.base = base
        self.bn = nn.BatchNorm1d(2048)

    def forward(self, x):
        return self.bn(self.base(x))


class ReIDEmbedder:
    INPUT_HW = (256, 128)

    def __init__(self, path, device):
        sd = _load_lightning_checkpoint(path)["state_dict"]
        self.model = CentroidReID()
        self.model.base.load_state_dict({k[len("backbone.base."):]: v for k, v in sd.items()
                                         if k.startswith("backbone.base.")}, strict=False)
        missing = [k for k in self.model.base.state_dict() if "backbone.base." + k not in sd and not k.startswith("fc")]
        if missing:
            raise ValueError(f"{path}: missing ReID weights, e.g. {missing[:3]}")
        self.model.bn.load_state_dict({k[3:]: v for k, v in sd.items() if k.startswith("bn.")})
        self.device = device
        self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, crops):
        h, w = self.INPUT_HW
        imgs = [cv2.cvtColor(cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB) for c in crops]
        return self.model(_to_tensor(imgs, IMAGENET_MEAN, IMAGENET_STD, self.device)).float().cpu().numpy()


def is_main_subject(features, index):
    """Their Gaussian outlier test (gaussian_outliers.get_main_subject) on one track's features:
    whether crop `index` survives OUTLIER_ROUNDS rounds. Tracks of two crops or fewer keep all."""
    if len(features) <= 2:
        return True
    kept = features
    for _ in range(OUTLIER_ROUNDS):
        mu = kept.mean(axis=0)
        dist = np.linalg.norm(features - mu, axis=1)
        inlier = (dist - dist.mean()) <= OUTLIER_THRESHOLD
        kept = features[inlier]
        if len(kept) == 0:
            return False
    return bool(inlier[index])


# ---------------------------------------------------------------- ViTPose-H

class _PatchEmbed(nn.Module):
    def __init__(self, patch=16, dim=1280):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, padding=2)  # ViTPose: 4 + 2 * (ratio // 2 - 1)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2), x.shape[2:]


class _Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1, self.attn, self.norm2, self.mlp = norm(dim), _Attention(dim, heads), norm(dim), _Mlp(dim, dim * 4)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _ViTBackbone(nn.Module):
    def __init__(self, dim=1280, depth=32, heads=16, tokens=192):
        super().__init__()
        self.patch_embed = _PatchEmbed(16, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, tokens + 1, dim))
        self.blocks = nn.ModuleList(_Block(dim, heads) for _ in range(depth))
        self.last_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        b = x.shape[0]
        x, (hp, wp) = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            x = blk(x)
        return self.last_norm(x).transpose(1, 2).reshape(b, -1, hp, wp)


class _HeatmapHead(nn.Module):
    def __init__(self, in_ch=1280, ch=256, joints=17):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_ch, ch, 4, 2, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.final_layer = nn.Conv2d(ch, joints, 1)

    def forward(self, x):
        return self.final_layer(self.deconv_layers(x))


class ViTPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _ViTBackbone()
        self.keypoint_head = _HeatmapHead()

    def forward(self, x):
        return self.keypoint_head(self.backbone(x))


class PoseEstimator:
    """ViTPose-H top-down: each player box is padded 1.25x to 3:4, warped to 256x192, and the 17
    COCO keypoints are read off the 64x48 heatmaps (mmpose's default decoding)."""
    INPUT_HW = (256, 192)
    BOX_PADDING = 1.25

    def __init__(self, path, device, half=False):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        self.model = ViTPose()
        self.model.load_state_dict(sd)
        self.device, self.half = device, half
        self.model.to(device).eval()
        if half:
            self.model.half()

    def _center_scale(self, box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        aspect = self.INPUT_HW[1] / self.INPUT_HW[0]
        if w > aspect * h:
            h = w / aspect
        else:
            w = h * aspect
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2]), np.array([w, h]) * self.BOX_PADDING

    @torch.no_grad()
    def __call__(self, frame, boxes):
        """(N, 17, 3) keypoints (x, y, score) in frame coordinates for N player boxes."""
        if not boxes:
            return np.zeros((0, 17, 3), np.float32)
        oh, ow = self.INPUT_HW
        crops, cs = [], []
        for box in boxes:
            c, s = self._center_scale(box)
            src = np.float32([c - s / 2, [c[0] + s[0] / 2, c[1] - s[1] / 2], [c[0] - s[0] / 2, c[1] + s[1] / 2]])
            dst = np.float32([[0, 0], [ow, 0], [0, oh]])
            warp = cv2.warpAffine(frame, cv2.getAffineTransform(src, dst), (ow, oh), flags=cv2.INTER_LINEAR)
            crops.append(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))
            cs.append((c, s))
        x = _to_tensor(crops, IMAGENET_MEAN, IMAGENET_STD, self.device)
        heat = self.model(x.half() if self.half else x).float().cpu().numpy()   # (N, 17, 64, 48)
        n, j, hh, hw = heat.shape
        flat = heat.reshape(n, j, -1)
        idx = flat.argmax(-1)
        score = flat.max(-1)
        px, py = (idx % hw).astype(np.float32), (idx // hw).astype(np.float32)
        # Quarter-pixel shift towards the higher neighbour (mmpose post_process='default').
        for b in range(n):
            for k in range(j):
                hm, ix, iy = heat[b, k], int(px[b, k]), int(py[b, k])
                if 0 < ix < hw - 1:
                    px[b, k] += 0.25 * np.sign(hm[iy, ix + 1] - hm[iy, ix - 1])
                if 0 < iy < hh - 1:
                    py[b, k] += 0.25 * np.sign(hm[iy + 1, ix] - hm[iy - 1, ix])
        out = np.zeros((n, j, 3), np.float32)
        for b, (c, s) in enumerate(cs):
            out[b, :, 0] = px[b] * s[0] / hw + c[0] - s[0] / 2
            out[b, :, 1] = py[b] * s[1] / hh + c[1] - s[1] / 2
            out[b, :, 2] = score[b]
        return out


def torso_from_pose(keypoints, frame_shape):
    """Their pose crop (helpers.generate_crops): shoulders to hips, padded 5 px except at the
    bottom, or None unless all four keypoints have confidence >= KEYPOINT_CONF."""
    pts = keypoints[[COCO_R_SHOULDER, COCO_L_SHOULDER, COCO_L_HIP, COCO_R_HIP]]
    if (pts[:, 2] < KEYPOINT_CONF).any():
        return None
    h, w = frame_shape[:2]
    x1 = max(0, int(pts[:, 0].min() - CROP_PADDING))
    y1 = max(0, int(pts[:, 1].min() - CROP_PADDING))
    x2 = int(min(w - 1, pts[:, 0].max() + CROP_PADDING))
    y2 = int(min(h - 1, pts[:, 1].max()))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


# ---------------------------------------------------------------- the pipeline

class PipelineReader:
    """ReID outlier filter -> pose torso crop -> PARSeq, per player track. Call reset() between videos."""
    votes = TrackletVotes   # how run_models.py combines a track's readings

    def __init__(self, models_dir, device):
        half = str(device).startswith("cuda") or str(device)[:1].isdigit()
        self.reid = ReIDEmbedder(models_dir / "centroid-reid.ckpt", device)
        self.pose = PoseEstimator(models_dir / "vitpose-h.pth", device, half=half)
        self.str = ParseqReader(models_dir / "jersey.ckpt", device)
        legibility = find_legibility_model(models_dir)
        self.legibility = LegibilityClassifier(legibility, device, half) if legibility else None
        self.stats = defaultdict(int)
        self.reset()

    def reset(self):
        self.features = defaultdict(list)

    def read(self, frame, boxes, keys):
        """One (number, confidence) or None per player box (xyxy) with its track key."""
        out = [None] * len(boxes)
        if not boxes:
            return out
        h, w = frame.shape[:2]
        crops = [frame[max(0, int(b[1])):min(h, int(b[3])), max(0, int(b[0])):min(w, int(b[2]))] for b in boxes]
        feats = self.reid(crops)
        main = []
        for i, (k, f) in enumerate(zip(keys, feats)):
            if k is None:
                main.append(i)
                continue
            hist = self.features[k]
            hist.append(f)
            del hist[:-MAX_TRACK_FEATURES]
            if is_main_subject(np.stack(hist), len(hist) - 1):
                main.append(i)
            else:
                self.stats["reid_outlier"] += 1
        self.stats["crops"] += len(boxes)
        if main and self.legibility is not None:
            legible = self.legibility([crops[i] for i in main])
            self.stats["illegible"] += sum(p <= LEGIBILITY_THRESHOLD for p in legible)
            main = [i for i, p in zip(main, legible) if p > LEGIBILITY_THRESHOLD]
        if not main:
            return out
        kpts = self.pose(frame, [boxes[i] for i in main])
        torsos, idx = [], []
        for i, kp in zip(main, kpts):
            t = torso_from_pose(kp, frame.shape)
            if t is None:
                self.stats["no_pose"] += 1
                continue
            torsos.append(frame[t[1]:t[3], t[0]:t[2]])
            idx.append(i)
        for i, (number, conf) in zip(idx, self.str.read(torsos) if torsos else []):
            if number is not None and not number.startswith("0"):
                out[i] = (number, conf)
                self.stats["read"] += 1
        return out


PipelineVotes = TrackletVotes   # their tracklet vote, now in parseq_jersey.py
