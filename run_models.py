"""Run the Hockey-Vision YOLO models on an image, a video, or a folder of either.

Examples:
    python run_models.py path/to/frame.jpg
    python run_models.py path/to/game.mp4 --models player puck --conf 0.3
    python run_models.py path/to/folder --no-per-model --output results

For every input this writes to <output>/<input name>/:
    combined.<ext>        all selected models drawn on one image/video
    <model>.<ext>         one annotated output per model (unless --no-per-model)
    detections.json       every detection (per frame for videos)
With --no-video only detections.json is written.

Each model is loaded from CV_Models/Models/<model>_model.engine when that exists (a TensorRT engine
built by export_engines.py, for the Jetson), otherwise from CV_Models/Models/<model>_model.pt; the player,
puck, dots, rink and number models are CV_Models/Models/new_player_model, new_nano_puck, new_dots,
new_rink_model and new_nums (.engine / .pt) instead (MODEL_FILES). The older jersey number readers'
models are in CV_Models/unused_models/jersey_Num Models/ (JERSEY_DIR). An engine
runs at the input size it was built for; --imgsz only applies to .pt weights, which run in FP16
on a CUDA GPU.

Jersey numbers are read by the YOLO number model (new_nums.pt): it detects single digits on each
player's box (upscaled) and joins them into the number. With --teams (team_a's and team_b's abbreviations, e.g. SJS MTL) a player's number can only
be one their team wears (roster.py, from a fetch_roster.py CSV, --roster), referees get none, and
the outputs name each numbered player. Older readers, kept for comparison (their models are in
JERSEY_DIR): --number-reader parseq (PARSeq, the text recognizer from Koshkina & Elder's jersey
pipeline, parseq_jersey.py), pipeline (their whole pipeline with a ReID filter and a ViTPose crop,
jersey_pipeline.py; ~10x the cost), resnet (jersey_net.py, colab/train_jersey.py) and temporal
(temporal_jersey.py). In videos players are tracked across frames, and each track's
number is a vote over every frame's reading, so it stays with the player through frames where it
can't be read and one-off misreads are outvoted.
Reading numbers is most of the GPU work per frame, so a player whose number is already settled is
only read every --jersey-stride frames (default 5; players without a number are read every frame).
On our test clips that reads a third fewer players with the same final numbers.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

from jersey_net import JerseyReader, checkpoint_path, torso_box, torso_crop
from jersey_pipeline import PipelineReader
from parseq_jersey import ParseqReader, find_legibility_model
from roster import DEFAULT_ROSTER, Roster
from temporal_jersey import TemporalJerseyReader
from video_io import FrameReader

MODELS_DIR = Path(__file__).parent / "CV_Models" / "Models"
# The older jersey number readers' models (--number-reader parseq / pipeline / resnet / temporal):
# PARSeq and its legibility classifier, the jersey ResNet, the temporal reader, the pipeline
# reader's Centroid-ReID and ViTPose-H, and the old YOLO digit model.
JERSEY_DIR = Path(__file__).parent / "CV_Models" / "unused_models" / "jersey_Num Models"
JERSEY_MODELS = {"jersey"}
MODEL_NAMES = ["player", "puck", "number", "rink", "dots"]

# BGR colour per model for the combined output, so each model's boxes are distinguishable.
MODEL_COLORS = {
    "player": (60, 180, 75),
    "puck": (0, 0, 255),
    "number": (0, 215, 255),
    "rink": (255, 130, 0),
    "dots": (240, 50, 230),
}

# Weights file stems (in CV_Models/Models/, before .pt / .engine) that aren't <model>_model.
MODEL_FILES = {
    "player": "new_player_model",
    "puck": "new_nano_puck",
    "dots": "new_dots",
    "rink": "new_rink_model",
    "number": "new_nums",
}

# Class names to rename in a model's output. new_nano_puck calls its one class "item"; the rest of
# the pipeline (homography.py, export_data.py) looks for "puck".
CLASS_RENAMES = {
    "puck": {"item": "puck"},
}

# Classes to drop from a model's output. The puck model handles pucks, so the player model's
# puck class is ignored to avoid duplicate boxes.
EXCLUDED_CLASSES = {
    "player": {"puck"},
}

# Per-model confidence thresholds that differ from --conf. new_nano_puck's boxes below about 0.6
# are often ad-board lettering, skates or gloves; above it they're nearly all pucks.
MODEL_CONF = {
    "puck": 0.60,
}

# Jersey numbers.
NUMBER_IMGSZ = 640          # player crops are upscaled to this for the digit model
# Player crops per digit-model call; export_engines.py builds the engine for up to this many.
# 16 covers every player in almost every frame in one call (~12% faster than 4, same readings). The
# old YOLO11m digit model needed more memory than an 8 GB Orin Nano has free to build a batch-16
# engine; if new_nums.pt's build runs out of memory, set this back to 4.
NUMBER_BATCH = 16
MIN_NUMBER_CROP_PX = 60     # players shorter than this are too small to read a number from
UNNUMBERED_CLASSES = {"referee"}  # player model classes the YOLO number reader skips
DIGIT_NMS_IOU = 0.5         # overlapping digit boxes of different classes: keep the most confident
# Jersey numbers sit on the torso: digit centres are 20-60% of the way down the player's box and
# digits are 6-30% of its height. Digits elsewhere are stripes on socks, sticks or the boards.
DIGIT_Y_RANGE = (0.1, 0.62)
DIGIT_HEIGHT_RANGE = (0.06, 0.3)
MIN_READING_CONF = 0.4      # weaker readings don't vote
TORSO_OVERLAP = 0.3         # jersey ResNet: skip a player when another player's box covers this much of their torso
MIN_JERSEY_VOTES = 1.2      # summed confidence a track's number needs before it is shown
PARTIAL_READ_WEIGHT = 0.5   # a one-digit read of a two-digit number (e.g. "2" of "12") counts this much
SWITCH_RATIO = 1.5          # a track's shown number only changes when another scores this much higher
# With --jersey-stride, a track's number counts as settled (and is read less often) once it has
# this much summed confidence and leads the next best number by SETTLED_LEAD times.
SETTLED_VOTES = 8.0
SETTLED_LEAD = 3.0

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def weights_stem(name):
    """The model's weights file name in CV_Models/Models/, without the .pt / .engine suffix."""
    return MODEL_FILES.get(name, f"{name}_model")


def model_dir(name):
    """The folder the model's weights and engine are in."""
    return JERSEY_DIR if name in JERSEY_MODELS else MODELS_DIR


def model_path(name):
    """The model's TensorRT engine if one has been exported, otherwise its .pt weights."""
    engine = model_dir(name) / f"{weights_stem(name)}.engine"
    if engine.exists():
        return engine
    return checkpoint_path(JERSEY_DIR) if name == "jersey" else model_dir(name) / f"{weights_stem(name)}.pt"


def fp16_kwargs(device):
    """predict() arguments for FP16 .pt inference on an NVIDIA GPU (engines have their own precision).
    Newer Ultralytics replaced `half` with `quantize`."""
    if not (str(device).startswith("cuda") or str(device)[:1].isdigit()):
        return {}
    return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}


def engine_imgsz(path):
    """The input size a TensorRT engine was built for ([h, w]), from the metadata Ultralytics
    writes at the start of the file, or None for .pt weights or an engine without it."""
    if path.suffix != ".engine":
        return None
    try:
        with open(path, "rb") as f:
            meta = json.loads(f.read(int.from_bytes(f.read(4), byteorder="little")).decode("utf-8"))
        return None if meta.get("dynamic") else meta.get("imgsz")
    except (ValueError, UnicodeDecodeError, AttributeError):
        return None


def load_model(name):
    path = model_path(name)
    if not path.exists():
        sys.exit(f"Model not found: {path}")
    return YOLO(str(path), task="detect")


def load_models(names, number_reader="yolo", device=None):
    """{name: model}. With number_reader "resnet", "number" is a JerseyReader instead of the YOLO
    digit model."""
    models = {}
    for name in names:
        if name == "number" and number_reader == "pipeline":
            needed = [JERSEY_DIR / n for n in ("centroid-reid.ckpt", "vitpose-h.pth", "jersey.ckpt")]
            for path in needed:
                if not path.exists():
                    sys.exit(f"Model not found: {path}")
            print(f"  {name}: " + " + ".join(p.name for p in needed))
            models[name] = PipelineReader(JERSEY_DIR, device)
        elif name == "number" and number_reader == "parseq":
            path = JERSEY_DIR / "jersey.ckpt"
            if not path.exists():
                sys.exit(f"Model not found: {path}")
            print(f"  {name}: {path.name}")
            legibility = find_legibility_model(JERSEY_DIR)
            print(f"  legibility: {legibility.name if legibility else 'none (every player is read)'}")
            models[name] = ParseqReader(path, device, legibility)
        elif name == "number" and number_reader == "temporal":
            path = JERSEY_DIR / "jersey_model.pth"
            if not path.exists():
                sys.exit(f"Model not found: {path}")
            print(f"  {name}: {path.name}")
            models[name] = TemporalJerseyReader(path, device)
        elif name == "number" and number_reader == "resnet":
            path = model_path("jersey")
            if not path.exists():
                sys.exit(f"Model not found: {path} (train it with colab/train_jersey.py)")
            print(f"  {name}: {path.name}")
            models[name] = JerseyReader(path, device)
        else:
            print(f"  {name}: {model_path(name).name}")
            models[name] = load_model(name)
    return models


def kept_classes(name, model):
    """Class ids to keep for a model, or None to keep all of them."""
    excluded = EXCLUDED_CLASSES.get(name)
    if not excluded:
        return None
    return [i for i, cls in model.names.items() if cls not in excluded]


def run_on_frame(models, frame, args, track=False):
    """Return {model_name: ultralytics Results} for a single BGR frame, on the CPU.

    Thresholds and input sizes come from args.model_conf and args.model_imgsz. The number model
    isn't run here: it reads digits off the player boxes, see read_jerseys(). With `track`,
    players get track ids that persist across calls (one video at a time).
    """
    out = {}
    for name, model in models.items():
        if name == "number":
            continue
        run = model.track if track and name == "player" else model.predict
        kwargs = {"persist": True, "tracker": "bytetrack.yaml"} if run == model.track else {}
        # Results come back on the GPU; one copy to the CPU per model beats a sync per .tolist().
        out[name] = run(frame, conf=args.model_conf[name], imgsz=args.model_imgsz[name], device=args.device,
                        classes=kept_classes(name, model), verbose=False, **args.fp16, **kwargs)[0].cpu()
        if name in CLASS_RENAMES:
            renames = CLASS_RENAMES[name]
            out[name].names = {i: renames.get(cls, cls) for i, cls in out[name].names.items()}
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def assemble_number(digits):
    """Join the digits found on one player into a jersey number. Returns (number, confidence)
    or None.

    Starts from the most confident digit and adds at most one neighbour: a digit of about the
    same height, level with it and next to it. Anything else in the box (a sleeve number, a
    stray read on a logo) is ignored.
    """
    kept = []
    for d in sorted(digits, key=lambda d: -d["confidence"]):
        if all(_iou(d["box_xyxy"], k["box_xyxy"]) < DIGIT_NMS_IOU for k in kept):
            kept.append(d)
    if not kept:
        return None
    best = kept[0]
    x1, y1, x2, y2 = best["box_xyxy"]
    h, cy = y2 - y1, (y1 + y2) / 2
    group = [best]
    for d in kept[1:]:
        dx1, dy1, dx2, dy2 = d["box_xyxy"]
        gap = max(dx1 - x2, x1 - dx2)  # horizontal space between the boxes, negative if they overlap
        if (abs((dy2 - dy1) - h) < 0.35 * h and abs((dy1 + dy2) / 2 - cy) < 0.5 * h
                and -0.3 * h < gap < 0.6 * h):
            group.append(d)
            break
    group.sort(key=lambda d: d["box_xyxy"][0])
    number = "".join(d["class"] for d in group)
    if number.startswith("0"):  # the NHL doesn't allow 0 or 00; a lone "0" is usually a logo
        return None
    return number, sum(d["confidence"] for d in group) / len(group)


def read_jerseys(number_model, frame, player_result, args, only=None, keys=None):
    """Read each player's number this frame. Returns one dict per player box, in box order:
    {"digits": digit detections in frame coordinates, "reading": (number, conf) or None,
    "number": the number to show, filled in by the caller}. With `only` (a set of box indices),
    the other players aren't read and get no digits or reading. `keys` are the players' track keys
    (only the temporal reader uses them). With args.roster (--teams), a reading can only be a
    number the player's team wears: PARSeq picks the likeliest of those, the other readers' readings
    of anything else are dropped."""
    roster = getattr(args, "roster", None)
    allowed = None
    if roster is not None:
        allowed = [roster.allowed(player_result.names[int(c)]) for c in player_result.boxes.cls.tolist()]
    if isinstance(number_model, (JerseyReader, ParseqReader)):
        return read_jerseys_resnet(number_model, frame, player_result, only, allowed)
    jerseys = _read_jerseys(number_model, frame, player_result, args, only, keys)
    if allowed is not None:
        for j, ok in zip(jerseys, allowed):
            if j["reading"] is not None and ok is not None and j["reading"][0] not in ok:
                j["reading"] = None
    return jerseys


def _read_jerseys(number_model, frame, player_result, args, only=None, keys=None):
    """read_jerseys() without the roster: the reader's own readings."""
    if isinstance(number_model, PipelineReader):
        return read_jerseys_pipeline(number_model, frame, player_result, keys, only)
    if isinstance(number_model, TemporalJerseyReader):
        return read_jerseys_temporal(number_model, frame, player_result, keys, only)
    jerseys = [{"digits": [], "reading": None, "number": None} for _ in range(len(player_result.boxes))]
    crops, offsets = [], []
    h, w = frame.shape[:2]
    classes = [player_result.names[int(c)] for c in player_result.boxes.cls.tolist()]
    for i, box in enumerate(player_result.boxes.xyxy.tolist()):
        x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
        if (only is not None and i not in only) or y2 - y1 < MIN_NUMBER_CROP_PX or x2 <= x1:
            continue
        if classes[i] in UNNUMBERED_CLASSES:  # referees only get detected so they aren't taken for players
            continue
        crops.append(frame[y1:y2, x1:x2])
        offsets.append((i, x1, y1))
    if not crops:
        return jerseys
    players = player_result.boxes.xyxy.tolist()
    results = []
    for start in range(0, len(crops), NUMBER_BATCH):
        results += [r.cpu() for r in number_model.predict(
            crops[start:start + NUMBER_BATCH], conf=args.model_conf["number"], imgsz=NUMBER_IMGSZ,
            device=args.device, verbose=False, **args.fp16)]
    for (i, ox, oy), r in zip(offsets, results):
        box_h = players[i][3] - players[i][1]
        for box, cls, score in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if not (DIGIT_Y_RANGE[0] <= (box[1] + box[3]) / 2 / box_h <= DIGIT_Y_RANGE[1]
                    and DIGIT_HEIGHT_RANGE[0] <= (box[3] - box[1]) / box_h <= DIGIT_HEIGHT_RANGE[1]):
                continue
            # Where players overlap, a digit inside both boxes could be either one's.
            cx, cy = (box[0] + box[2]) / 2 + ox, (box[1] + box[3]) / 2 + oy
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for k, b in enumerate(players) if k != i):
                continue
            jerseys[i]["digits"].append({
                "class": r.names[int(cls)],
                "confidence": round(score, 4),
                "box_xyxy": [round(v + o, 1) for v, o in zip(box, (ox, oy, ox, oy))],
            })
        jerseys[i]["reading"] = assemble_number(jerseys[i]["digits"])
    return jerseys


def torso_covered(i, players):
    """True when another player's box covers much of player i's torso, so a number read there could be theirs."""
    tx1, ty1, tx2, ty2 = torso_box(players[i])
    area = max(1e-6, (tx2 - tx1) * (ty2 - ty1))
    return any(max(0, min(tx2, b[2]) - max(tx1, b[0])) * max(0, min(ty2, b[3]) - max(ty1, b[1])) / area
               > TORSO_OVERLAP for k, b in enumerate(players) if k != i)


def read_jerseys_pipeline(reader, frame, player_result, keys=None, only=None):
    """read_jerseys() with Koshkina & Elder's pipeline (jersey_pipeline.py): every player's crop is
    checked against their track by ReID, cropped by pose and read by PARSeq."""
    jerseys = [{"digits": [], "reading": None, "number": None} for _ in range(len(player_result.boxes))]
    players = player_result.boxes.xyxy.tolist()
    keys = keys or [None] * len(players)
    read = [i for i, box in enumerate(players) if (only is None or i in only)
            and box[3] - box[1] >= MIN_NUMBER_CROP_PX and not torso_covered(i, players)]
    for i, reading in zip(read, reader.read(frame, [players[i] for i in read], [keys[i] for i in read])):
        jerseys[i]["reading"] = reading
    return jerseys


def read_jerseys_temporal(reader, frame, player_result, keys=None, only=None):
    """read_jerseys() with the temporal (EfficientNet + LSTM) model: each player's whole-box crop
    joins their track's sequence, and the sequence is read. Called once per frame, even with no
    players, so the reader keeps count of frames."""
    jerseys = [{"digits": [], "reading": None, "number": None} for _ in range(len(player_result.boxes))]
    players = player_result.boxes.xyxy.tolist()
    keys = keys or [None] * len(players)
    h, w = frame.shape[:2]
    crops, read = [], []
    for i, box in enumerate(players):
        if (only is not None and i not in only) or box[3] - box[1] < MIN_NUMBER_CROP_PX or torso_covered(i, players):
            continue
        x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            crops.append(frame[y1:y2, x1:x2])
            read.append(i)
    for i, reading in zip(read, reader.read(crops, [keys[i] for i in read])):
        if reading is not None and not reading[0].startswith("0"):
            jerseys[i]["reading"] = reading
    return jerseys


def read_jerseys_resnet(reader, frame, player_result, only=None, allowed=None):
    """read_jerseys() with the jersey ResNet (or PARSeq): one whole-number reading per player's
    torso crop, and no digit boxes. A reader with its own crop() (PARSeq) cuts the crop itself."""
    jerseys = [{"digits": [], "reading": None, "number": None} for _ in range(len(player_result.boxes))]
    players = player_result.boxes.xyxy.tolist()
    if hasattr(reader, "read_boxes"):   # PARSeq: legibility check on the whole player, then its own crop
        read = [i for i, box in enumerate(players) if (only is None or i in only)
                and box[3] - box[1] >= MIN_NUMBER_CROP_PX and not torso_covered(i, players)]
        sub = None if allowed is None else [allowed[i] for i in read]
        for i, (number, conf) in zip(read, reader.read_boxes(frame, [players[i] for i in read], sub)):
            if number is not None and not number.startswith("0"):
                jerseys[i]["reading"] = (number, conf)
        return jerseys
    crops, read = [], []
    for i, box in enumerate(players):
        if (only is not None and i not in only) or box[3] - box[1] < MIN_NUMBER_CROP_PX:
            continue
        if torso_covered(i, players):
            continue
        crop = reader.crop(frame, box) if hasattr(reader, "crop") else torso_crop(frame, box)
        if crop.shape[0] >= 8 and crop.shape[1] >= 8:
            crops.append(crop)
            read.append(i)
    for i, (number, conf) in zip(read, reader.read(crops) if crops else []):
        if number is not None and not number.startswith("0"):
            jerseys[i]["reading"] = (number, conf)
    return jerseys


class JerseyVotes:
    """Per player track (any hashable key), the summed confidence of every number read on it. Once a track's number
    is shown it sticks to the player, whether or not the number is readable in later frames."""

    def __init__(self):
        self.votes = defaultdict(Counter)
        self.shown = {}

    def add(self, track, reading):
        if track is not None and reading is not None and reading[1] >= MIN_READING_CONF:
            self.votes[track][reading[0]] += reading[1]

    @staticmethod
    def _score(votes, n):
        # A two-digit number is also backed by reads that only caught one of its digits.
        partial = sum(votes[d] for d in set(n) if d in votes) if len(n) == 2 else 0.0
        return votes[n] + PARTIAL_READ_WEIGHT * partial

    def settled(self, track):
        """True once the shown number is so far ahead that more reads are very unlikely to change it."""
        shown = self.shown.get(track)
        if shown is None:
            return False
        votes = self.votes[track]
        top = self._score(votes, shown)
        # Digits of the shown number back it rather than compete with it.
        rivals = [self._score(votes, n) for n in votes if n != shown and not (len(shown) == 2 and n in shown)]
        return top >= SETTLED_VOTES and top >= SETTLED_LEAD * max(rivals, default=0.0)

    def number(self, track):
        votes = self.votes.get(track)
        if not votes:
            return None

        def score(n):
            return self._score(votes, n)

        best = max(votes, key=score)
        current = self.shown.get(track)
        if current is not None and score(best) < SWITCH_RATIO * score(current):
            return current
        if score(best) >= MIN_JERSEY_VOTES:
            self.shown[track] = best
        return self.shown.get(track)


def track_ids(result):
    ids = result.boxes.id
    return [None] * len(result.boxes) if ids is None else [int(i) for i in ids.tolist()]


def results_to_dicts(results, jerseys=None, roster=None):
    out = {}
    for name, r in results.items():
        dets = []
        ids = track_ids(r)
        for i, (box, cls, score) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                                   r.boxes.conf.tolist())):
            det = {
                "class": r.names[int(cls)],
                "confidence": round(score, 4),
                "box_xyxy": [round(v, 1) for v in box],
            }
            if name == "player":
                if ids[i] is not None:
                    det["track_id"] = ids[i]
                if jerseys is not None:
                    reading = jerseys[i]["reading"]
                    det["jersey_number"] = jerseys[i]["number"]
                    if roster is not None:   # the team from the class; the name once the number is known
                        det["team"], det["player_name"] = roster.player(det["class"], det["jersey_number"])
                    det["jersey_reading"] = None if reading is None else {
                        "number": reading[0], "confidence": round(reading[1], 4)}
            dets.append(det)
        out[name] = dets
    if jerseys is not None:
        out["number"] = [d for j in jerseys for d in j["digits"]]
    return out


def draw_label(img, text, x, y, color):
    scale = max(0.4, img.shape[1] / 2500)
    thickness = max(1, int(scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y = max(y, th + baseline)
    cv2.rectangle(img, (x, y - th - baseline), (x + tw, y), color, -1)
    cv2.putText(img, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def draw_jersey(img, number, box, color):
    """A "#12" badge centred above the player's box (above its class label), clear of the jersey."""
    x1, y1, x2, _ = box
    text = f"#{number}"
    scale = max(0.6, img.shape[1] / 1600)
    thickness = max(1, int(scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    label_scale = max(0.4, img.shape[1] / 2500)  # same as draw_label
    (_, lh), lb = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, label_scale, max(1, int(label_scale * 2)))
    label_h = lh + lb
    x = int((x1 + x2) / 2 - tw / 2)
    y = max(th + baseline + 4, int(y1) - label_h - 4)
    cv2.rectangle(img, (x - 4, y - th - baseline - 4), (x + tw + 4, y + 2), color, -1)
    cv2.rectangle(img, (x - 4, y - th - baseline - 4), (x + tw + 4, y + 2), (255, 255, 255), 1)
    cv2.putText(img, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def draw_numbers(frame, results, jerseys, digits=True):
    """Digit boxes and each player's jersey number badge."""
    img = frame.copy()
    color = MODEL_COLORS["number"]
    line = max(1, int(img.shape[1] / 800))
    for box, j in zip(results["player"].boxes.xyxy.tolist(), jerseys):
        if digits:
            for d in j["digits"]:
                x1, y1, x2, y2 = map(int, d["box_xyxy"])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, line)
        if j["number"] is not None:
            draw_jersey(img, j["number"], box, MODEL_COLORS["player"])
    return img


def draw_combined(frame, results, jerseys=None):
    img = frame.copy()
    line = max(1, int(img.shape[1] / 800))
    for name, r in results.items():
        color = MODEL_COLORS.get(name, (200, 200, 200))
        ids = track_ids(r)
        for box, cls, score, tid in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                        r.boxes.conf.tolist(), ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line)
            label = f"{r.names[int(cls)]} {score:.2f}" + (f" id{tid}" if tid is not None else "")
            draw_label(img, label, x1, y1, color)
    if jerseys is not None:
        img = draw_numbers(img, results, jerseys)

    # Legend in the top-left corner.
    y = 10
    for name in list(results) + (["number"] if jerseys is not None else []):
        y += 25
        draw_label(img, name, 10, y, MODEL_COLORS.get(name, (200, 200, 200)))
    return img


def per_model_frame(name, frame, results, jerseys):
    if name == "number":
        return draw_numbers(frame, results, jerseys)
    return results[name].plot()


def process_image(path, models, args, out_dir):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"  Could not read {path}, skipping")
        return
    results = run_on_frame(models, frame, args)
    jerseys = None
    if "number" in models:
        # A single image has nothing to vote over, so each player shows this frame's reading.
        jerseys = read_jerseys(models["number"], frame, results["player"], args)
        for j in jerseys:
            j["number"] = j["reading"][0] if j["reading"] else None

    if args.write_video:
        cv2.imwrite(str(out_dir / f"combined{path.suffix}"), draw_combined(frame, results, jerseys))
    if args.write_video and args.per_model:
        for name in models:
            cv2.imwrite(str(out_dir / f"{name}{path.suffix}"), per_model_frame(name, frame, results, jerseys))

    detections = results_to_dicts(results, jerseys, args.roster)
    (out_dir / "detections.json").write_text(json.dumps(detections, indent=2))
    counts = ", ".join(f"{n}: {len(d)}" for n, d in detections.items())
    print(f"  {counts}")


def process_video(path, models, args, out_dir):
    try:
        video = FrameReader(path, args.max_frames)
    except OSError:
        print(f"  Could not open {path}, skipping")
        return
    fps, w, h, total = video.fps, video.width, video.height, video.total

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {}
    if args.write_video:
        writers["combined"] = cv2.VideoWriter(str(out_dir / "combined.mp4"), fourcc, fps, (w, h))
        if args.per_model:
            for name in models:
                writers[name] = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), fourcc, fps, (w, h))

    if "player" in models:
        # A fresh player model per video, so track ids (and the tracker's state) start over.
        models = dict(models, player=load_model("player"))
    # A reader may combine a track's readings its own way (PARSeq: Koshkina & Elder's tracklet vote).
    votes = getattr(models.get("number"), "votes", JerseyVotes)()
    if hasattr(models.get("number"), "reset"):
        models["number"].reset()  # the temporal reader's per-track sequences start over

    # Frames are written to detections.json as they're done (one per line) rather than held in
    # memory, which a full game would fill on the Jetson. The array is closed even if the run is
    # interrupted, so the frames done so far are still usable.
    detections_file = open(out_dir / "detections.json", "w")
    detections_file.write("[")
    idx = 0
    try:
        for frame in video:
            results = run_on_frame(models, frame, args, track=True)
            jerseys = None
            if "number" in models:
                # The tracker sometimes hands an id over to a nearby player; keying on the class
                # too means a handover to the other team (or a referee) doesn't inherit the number.
                r = results["player"]
                keys = [None if tid is None else (tid, r.names[int(cls)])
                        for tid, cls in zip(track_ids(r), r.boxes.cls.tolist())]
                # Players whose number is settled are only read every jersey_stride frames,
                # staggered by track id so the reads are spread evenly over the frames.
                only = None
                if args.jersey_stride > 1:
                    only = {i for i, key in enumerate(keys) if key is None or not votes.settled(key)
                            or (idx + key[0]) % args.jersey_stride == 0}
                jerseys = read_jerseys(models["number"], frame, r, args, only, keys)
                for key, j in zip(keys, jerseys):
                    votes.add(key, j["reading"])
                for key, j in zip(keys, jerseys):
                    j["number"] = votes.number(key)

            if "combined" in writers:
                writers["combined"].write(draw_combined(frame, results, jerseys))
            for name in models:
                if name in writers:
                    writers[name].write(per_model_frame(name, frame, results, jerseys))

            detections_file.write(("\n" if idx == 0 else ",\n") + json.dumps(
                {"frame": idx, "time_s": round(idx / fps, 3), "detections": results_to_dicts(results, jerseys, args.roster)}))
            idx += 1
            if idx % 25 == 0 or idx == total:
                print(f"\r  frame {idx}/{total or '?'}", end="", flush=True)
    finally:
        detections_file.write("\n]\n")
        detections_file.close()
        video.close()
        for wr in writers.values():
            wr.release()
    print()


def collect_inputs(source):
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
    return [source]


def main():
    parser = argparse.ArgumentParser(description="Run the Hockey-Vision models and save annotated outputs.")
    parser.add_argument("source", type=Path, help="Image, video, or folder of images/videos")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES,
                        help="Which models to run (default: all)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for every model except the puck (default: 0.25)")
    parser.add_argument("--puck-conf", type=float, default=MODEL_CONF["puck"],
                        help=f"Confidence threshold for the puck model (default: {MODEL_CONF['puck']})")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--device", default=None, help="cpu, mps, cuda, 0... (default: auto)")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output folder (default: outputs)")
    parser.add_argument("--no-per-model", dest="per_model", action="store_false",
                        help="Only write the combined output, not one per model")
    parser.add_argument("--no-video", dest="write_video", action="store_false",
                        help="Only write detections.json, no annotated images/videos (faster)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop videos after N frames (0 = all)")
    parser.add_argument("--number-reader", choices=["yolo", "parseq", "pipeline", "resnet", "temporal"], default="yolo",
                        help="How jersey numbers are read (default: yolo, the new_nums.pt digit detector); "
                             "parseq, pipeline, resnet and temporal are older readers kept for comparison, "
                             "with their models in CV_Models/unused_models/jersey_Num Models/")
    parser.add_argument("--teams", nargs=2, metavar=("TEAM_A", "TEAM_B"),
                        help="Abbreviations of the teams the player model calls team_a and team_b (e.g. SJS MTL): "
                             "jersey numbers are then limited to their rosters and players are named")
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER,
                        help="Roster CSV from fetch_roster.py, for --teams (default: rosters/nhl_active_players.csv; "
                             "for a past game, its own: python fetch_roster.py --game <id>)")
    parser.add_argument("--jersey-stride", type=int, default=5,
                        help="In videos, re-read a player's jersey number only every N frames once it is "
                             "settled (default: 5; players without a settled number are read every frame; "
                             "1 reads every player every frame)")
    args = parser.parse_args()

    if not args.source.exists():
        sys.exit(f"Source not found: {args.source}")
    args.device = pick_device(args.device)
    args.fp16 = fp16_kwargs(args.device)
    args.roster = Roster(args.roster, args.teams) if args.teams else None
    if args.roster:
        print(f"Rosters from {args.roster.path.name}: {args.roster.describe()}")
    args.model_conf = {name: MODEL_CONF.get(name, args.conf) for name in MODEL_NAMES}
    args.model_conf["puck"] = args.puck_conf

    if "number" in args.models and "player" not in args.models:
        args.models = args.models + ["player"]  # jersey numbers are read off the player boxes
        print("Adding the player model: the number model needs player boxes")

    inputs = collect_inputs(args.source)
    if not inputs:
        sys.exit(f"No images or videos found in {args.source}")

    print(f"Loading models: {', '.join(args.models)} (device: {args.device})")
    models = load_models(args.models, args.number_reader, args.device)
    args.model_imgsz = {name: engine_imgsz(model_path(name)) or args.imgsz for name in args.models}

    for path in inputs:
        out_dir = args.output / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{path} -> {out_dir}/")
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            process_image(path, models, args, out_dir)
        elif ext in VIDEO_EXTS:
            process_video(path, models, args, out_dir)
        else:
            print(f"  Unsupported file type {ext}, skipping")


if __name__ == "__main__":
    main()
