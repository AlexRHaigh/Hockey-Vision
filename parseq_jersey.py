"""Jersey numbers read as text by PARSeq, the scene-text recognizer from Koshkina & Elder's jersey
number pipeline (github.com/mkoshkina/jersey-number-pipeline), from CV_Models/unused_models/jersey_Num Models/jersey.ckpt. This is
run_models.py's default jersey reader (--number-reader parseq).

Per player, their legibility classifier (CV_Models/unused_models/jersey_Num Models/legibility_resnet34_hockey_*.pth) first decides
from the whole player crop whether a number is readable at all; PARSeq only reads players it
passes. Without it PARSeq reads every player, and on blurred or turned players it tends to answer
"4" with high confidence. PARSeq reads a fixed crop of the player's box (CROP_BOX): where their
pipeline's pose model (ViTPose) puts the shoulders-to-hips torso, as measured on ~1,000 players
from our footage.
That keeps most of the pose crop's benefit without the pose model, which costs ~250 GFLOPs per
player. jersey_pipeline.py runs their full pipeline (ReID filter, ViTPose crop) for comparison.

As in their str.py, PARSeq's output is restricted to digits (the first 3 positions, and only the
end-of-text and 0-9 tokens), and a reading counts when it is one or two digits. Its confidence is
the product of the digits' probabilities (the end-of-text probability left out, as their
consolidation does). With the teams' rosters (run_models.py --teams), a reading is instead the
allowed number (that team's numbers, see roster.py) PARSeq's digit probabilities score highest:
an impossible "4" or a dropped digit ("7" for 72) gives way to a number someone actually wears.
Each track's readings are combined with their vote, TrackletVotes.

Needs the PARSeq model code: pip install --no-deps git+https://github.com/baudm/parseq.git (plus
timm). The checkpoint (epoch 3, step 95) is their hockey fine-tune: their configuration.py lists it
for Hockey; their soccer one is epoch 24, step 2575.
"""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

# Crop PARSeq reads, as fractions of the player box (x1, y1, x2, y2): the median ViTPose
# shoulders-to-hips torso (padded 5 px, as their pipeline crops it) over 988 players from 10 s
# of our footage 30 minutes into the game.
CROP_BOX = (0.22, 0.15, 0.78, 0.45)
MAX_BATCH = 16
MAX_CHARS = 3            # positions decoded: up to two digits and the end-of-text token
DIGIT_TOKENS = 11        # end-of-text, then 0-9: the first 11 of PARSeq's output classes

LEGIBILITY_THRESHOLD = 0.5   # their legibility_classifier.run
LEGIBILITY_HW = (256, 256)   # their test transform: the whole player crop, squashed
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Their tracklet vote (helpers.process_jersey_id_predictions with useBias=True)
FILTER_THRESHOLD = 0.2   # weaker readings count for nothing
SUM_THRESHOLD = 1.0      # summed weight a number needs
TWO_DIGIT_WEIGHT, ONE_DIGIT_WEIGHT = 0.61, 0.39
SETTLED_WEIGHT, SETTLED_LEAD = 3.0, 3.0   # ours, for --jersey-stride (TrackletVotes.settled)


class TrackletVotes:
    """Koshkina & Elder's per-tracklet vote, with the same interface as run_models.JerseyVotes:
    each reading adds its confidence (0 under FILTER_THRESHOLD) times a bias towards two-digit
    numbers, and the best number is shown once its summed weight passes SUM_THRESHOLD."""

    def __init__(self):
        self.weights = defaultdict(lambda: defaultdict(float))

    def add(self, track, reading):
        if track is None or reading is None:
            return
        number, conf = reading
        conf = conf if conf >= FILTER_THRESHOLD else 0.0
        self.weights[track][number] += conf * (TWO_DIGIT_WEIGHT if len(number) == 2 else ONE_DIGIT_WEIGHT)

    def number(self, track):
        w = self.weights.get(track)
        if not w:
            return None
        best = max(w, key=w.get)
        return best if w[best] > SUM_THRESHOLD else None

    def settled(self, track):
        """For --jersey-stride: the track's number is far enough ahead that more reads can be
        spared. Their pipeline reads every image of a tracklet."""
        w = self.weights.get(track)
        if not w:
            return False
        ranked = sorted(w.values(), reverse=True)
        return ranked[0] >= SETTLED_WEIGHT and ranked[0] >= SETTLED_LEAD * (ranked[1] if len(ranked) > 1 else 0.0)


def find_legibility_model(models_dir):
    """Their hockey legibility classifier in models_dir, or None."""
    found = sorted(Path(models_dir).glob("legibility*.pth"))
    return found[-1] if found else None


class LegibilityClassifier:
    """Koshkina & Elder's legibility classifier (networks.LegibilityClassifier34): ResNet-34 with
    one sigmoid output, on the whole player crop squashed to 256x256. Above 0.5: a number is readable."""

    def __init__(self, path, device, half=False):
        model = torchvision.models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict({k.removeprefix("model_ft."): v for k, v in state.items()})
        self.model, self.device, self.half = model.to(device).eval(), device, half
        if half:
            self.model.half()

    @torch.no_grad()
    def __call__(self, crops):
        """Probability that each BGR player crop shows a readable number."""
        h, w = LEGIBILITY_HW
        imgs = [cv2.cvtColor(cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB) for c in crops]
        x = torch.from_numpy(np.stack(imgs)).float().div_(255)
        x = ((x - torch.from_numpy(IMAGENET_MEAN)) / torch.from_numpy(IMAGENET_STD)).permute(0, 3, 1, 2).contiguous()
        x = x.to(self.device)
        out = []
        for start in range(0, len(x), MAX_BATCH):
            b = x[start:start + MAX_BATCH]
            out += torch.sigmoid(self.model(b.half() if self.half else b).float()).flatten().tolist()
        return out


class ParseqReader:
    votes = TrackletVotes   # how run_models.py combines a track's readings

    def __init__(self, path, device, legibility_path=None):
        try:
            from strhub.data.utils import Tokenizer
            from strhub.models.parseq.model import PARSeq
        except ImportError as e:
            raise SystemExit("The PARSeq reader needs its model code: "
                             "pip install --no-deps git+https://github.com/baudm/parseq.git (and timm)") from e

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hp = ckpt["hyper_parameters"]
        self.tokenizer = Tokenizer(hp["charset_train"])
        self.model = PARSeq(len(self.tokenizer), hp["max_label_length"], hp["img_size"], hp["patch_size"],
                            hp["embed_dim"], hp["enc_num_heads"], hp["enc_mlp_ratio"], hp["enc_depth"],
                            hp["dec_num_heads"], hp["dec_mlp_ratio"], hp["dec_depth"], hp["decode_ar"],
                            hp["refine_iters"], hp["dropout"])
        state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
        self.model.load_state_dict(state)
        self.input_hw = tuple(hp["img_size"])   # (32, 128)
        self.device = device
        # FP16 on an NVIDIA GPU: on our test crops it read the same number as FP32 on 99.6% of them.
        self.half = str(device).startswith("cuda") or str(device)[:1].isdigit()
        self.model.to(device).eval()
        if self.half:
            self.model.half()
        self.crop_box = CROP_BOX
        self.legibility = LegibilityClassifier(legibility_path, device, self.half) if legibility_path else None

    def crop(self, img, box):
        """The part of a player box PARSeq reads (CROP_BOX), cut out of a BGR image."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        fx1, fy1, fx2, fy2 = self.crop_box
        return img[max(0, int(y1 + fy1 * bh)):min(h, int(y1 + fy2 * bh)),
                   max(0, int(x1 + fx1 * bw)):min(w, int(x1 + fx2 * bw))]

    def _prepare(self, crops):
        h, w = self.input_hw
        # As PARSeq's own transform: squash to 32x128 (bicubic), scale to [-1, 1].
        imgs = [cv2.cvtColor(cv2.resize(c, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB) for c in crops]
        x = torch.from_numpy(np.stack(imgs)).float().div_(127.5).sub_(1.0)
        x = x.permute(0, 3, 1, 2).contiguous().to(self.device)
        return x.half() if self.half else x

    def read_boxes(self, frame, boxes, allowed=None):
        """One (number or None, confidence) per player box (xyxy) in a BGR frame: the legibility
        classifier on the whole player crop, then PARSeq on the CROP_BOX crop of the legible ones.
        `allowed`: per box, the numbers it can be (see read()), or None."""
        out = [(None, 0.0)] * len(boxes)
        h, w = frame.shape[:2]
        keep = list(range(len(boxes)))
        if self.legibility is not None and boxes:
            players = [frame[max(0, int(b[1])):min(h, int(b[3])), max(0, int(b[0])):min(w, int(b[2]))] for b in boxes]
            keep = [i for i, p in zip(keep, self.legibility(players)) if p > LEGIBILITY_THRESHOLD]
        crops = [self.crop(frame, boxes[i]) for i in keep]
        ok = [(i, c) for i, c in zip(keep, crops) if c.shape[0] >= 8 and c.shape[1] >= 8]
        sub = None if allowed is None else [allowed[i] for i, _ in ok]
        for (i, _), reading in zip(ok, self.read([c for _, c in ok], sub) if ok else []):
            out[i] = reading
        return out

    def _score(self, dist, number):
        """Probability of reading exactly `number`: each digit at its position, then end-of-text.
        `dist` is the (MAX_CHARS, DIGIT_TOKENS) softmax for one crop."""
        p = 1.0
        for i, ch in enumerate(number):
            p *= float(dist[i, self.tokenizer._stoi[ch]])
        return p * float(dist[len(number), self.tokenizer.eos_id])

    @torch.no_grad()
    def read(self, crops, allowed=None):
        """One (number or None, confidence) per BGR crop. `allowed`, if given, is per crop a list of
        the numbers it can be (the most probable of them is read; an empty list reads nothing) or
        None (read freely)."""
        readings = []
        for start in range(0, len(crops), MAX_BATCH):
            logits = self.model(self.tokenizer, self._prepare(crops[start:start + MAX_BATCH]), MAX_CHARS)
            dists = logits[:, :MAX_CHARS, :DIGIT_TOKENS].float().softmax(-1)
            labels, probs = self.tokenizer.decode(dists)
            for j, (text, p) in enumerate(zip(labels, probs)):
                options = None if allowed is None else allowed[start + j]
                if options is not None:
                    scored = [(self._score(dists[j], n), n) for n in options]
                    best = max(scored) if scored else (0.0, None)
                    readings.append((best[1], best[0]) if best[1] is not None else (None, 0.0))
                    continue
                conf = float(p[:-1].prod()) if len(text) < len(p) else float(p.prod())  # without end-of-text
                readings.append((text, conf) if text.isdigit() and 1 <= len(text) <= 2 else (None, conf))
        return readings
