"""Turn a YOLO jersey dataset (e.g. a Roboflow YOLO export) into whole-number labels for
train_jersey.py.

    python colab/make_jersey_labels.py path/to/dataset            # images are whole-player crops
    python colab/make_jersey_labels.py path/to/dataset --torso    # images are already torso crops

The dataset folder needs data.yaml and images with YOLO .txt labels, in either layout:
<split>/images + <split>/labels (Roboflow) or images/<split> + labels/<split> (Ultralytics). It
writes <dataset>/labels.csv with columns path, number, split; number is empty for an image with no
number, which teaches the model to say "no readable number".

Two kinds of dataset work, told apart by the class names:
    digits         classes are single digits ("0".."9", or e.g. "digit-7"), one box per digit,
                   like the data number_model.pt was trained on: the digits in an image are
                   joined left to right into its number
    whole numbers  any class is a two-digit number ("23", "number-23"), one box per number
Digits named as words ("zero", "cero") count as digits; classes that aren't numbers (e.g.
"jersey", "player") are ignored. Polygon labels work too.

Images with more than two digits (a sleeve number as well as the back one), two different whole
numbers, or a leading zero are skipped. For whole-player crops, images whose number falls outside
the torso band the model sees (jersey_net.TORSO_Y) are skipped too, since it wouldn't be in its crop.

Splits come from folder names (train / valid / val / test). Keep each game in one split: frames
of the same player in both train and validation make validation accuracy meaningless. Roboflow
splits images at random, so this warns when the same source clip seems to be in several splits.
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

# jersey_net.py is at the repo root (run_models.py uses it too); on Colab it sits next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jersey_net import TORSO_Y

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
# Digit classes named as words (Roboflow datasets mix labellers and languages, e.g. "cero").
DIGIT_WORDS = {w: str(i) for i, words in enumerate([
    ("zero", "cero"), ("one", "uno"), ("two", "dos"), ("three", "tres"), ("four", "cuatro"),
    ("five", "cinco"), ("six", "seis"), ("seven", "siete"), ("eight", "ocho"), ("nine", "nueve"),
]) for w in words}


def class_numbers(data_yaml):
    """The number each class stands for ("7", "23"), or None for classes that aren't numbers."""
    names = yaml.safe_load(data_yaml.read_text())["names"]
    names = names if isinstance(names, list) else [names[i] for i in sorted(names)]
    numbers = []
    for n in names:
        found = re.findall(r"\d+", str(n))
        word = DIGIT_WORDS.get(re.sub(r"[^a-z]", "", str(n).lower()))
        numbers.append(found[0] if len(found) == 1 and len(found[0]) <= 2 else word)
    words = [f"{n!r} -> {num}" for n, num in zip(names, numbers) if num is not None and not re.search(r"\d", str(n))]
    if words:
        print(f"Reading classes named as words as digits: {', '.join(words)}")
    ignored = [str(n) for n, num in zip(names, numbers) if num is None]
    if ignored:
        print(f"Ignoring classes that aren't numbers: {', '.join(ignored)}")
    if not any(numbers):
        sys.exit(f"No number classes in {data_yaml}")
    return numbers


def box_centre(row):
    """(x, y) centre of a YOLO label row: a box (class x y w h) or a polygon (class x1 y1 x2 y2 ...)."""
    values = [float(v) for v in row[1:]]
    if len(values) == 4:
        return values[0], values[1]
    xs, ys = values[0::2], values[1::2]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def source_clip(img):
    """A guess at the clip an image came from: its original file name (Roboflow appends
    "_jpg.rf.<hash>") without the trailing frame number."""
    name = re.split(r"_(?:jpe?g|png|bmp|webp)\.rf\.", img.name, flags=re.IGNORECASE)[0]
    return re.sub(r"[\d_\-. ]+$", "", Path(name).stem) or name


def label_path(img):
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")  # the last "images" folder in the path
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def main():
    parser = argparse.ArgumentParser(description="Make whole-number jersey labels from a YOLO digit dataset.")
    parser.add_argument("dataset", type=Path, help="Folder containing data.yaml")
    parser.add_argument("--torso", action="store_true",
                        help="Images are already torso crops (default: whole-player crops)")
    args = parser.parse_args()

    numbers = class_numbers(args.dataset / "data.yaml")
    whole_numbers = any(n is not None and len(n) == 2 for n in numbers)
    print("Classes are whole numbers (one box per number)" if whole_numbers
          else "Classes are digits (one box per digit, joined left to right)")
    images = sorted(p for p in args.dataset.rglob("*") if p.suffix.lower() in IMAGE_EXTS and "images" in p.parts)
    if not images:
        sys.exit(f"No images under an images/ folder in {args.dataset}")

    skipped = Counter()
    rows = []
    wide = 0  # images wider than tall: full frames rather than player crops?
    clips = {}
    for img in images:
        split = next((SPLITS[p] for p in img.relative_to(args.dataset).parts if p in SPLITS), None)
        if split is None:
            skipped["no train/valid/test folder in its path"] += 1
            continue
        lbl = label_path(img)
        rows_txt = [line.split() for line in lbl.read_text().splitlines() if line.strip()] if lbl.exists() else []
        # (x, y, number) per box, coordinates 0-1, left to right; classes that aren't numbers are dropped.
        boxes = sorted((*box_centre(r), numbers[int(r[0])]) for r in rows_txt if numbers[int(r[0])] is not None)
        if whole_numbers:
            if len({n for _, _, n in boxes}) > 1:
                skipped["two different numbers"] += 1
                continue
            number = boxes[0][2] if boxes else ""
        else:
            if len(boxes) > 2:
                skipped["more than two digits"] += 1
                continue
            number = "".join(n for _, _, n in boxes)
        if not args.torso and any(not TORSO_Y[0] <= y <= TORSO_Y[1] for _, y, _ in boxes):
            skipped["number outside the torso crop"] += 1
            continue
        if number.startswith("0"):
            skipped["leading zero"] += 1
            continue
        rows.append([img.relative_to(args.dataset).as_posix(), number, split])
        clips.setdefault(source_clip(img), set()).add(split)
        if not args.torso:
            with Image.open(img) as im:
                wide += im.width > im.height

    out = args.dataset / "labels.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "number", "split"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} images to {out}")
    for split in ("train", "valid", "test"):
        in_split = [r for r in rows if r[2] == split]
        if in_split:
            none = sum(not r[1] for r in in_split)
            print(f"  {split}: {len(in_split)} images, {len({r[1] for r in in_split if r[1]})} different numbers, "
                  f"{none} ({none / len(in_split):.0%}) with no number")
    for reason, n in skipped.most_common():
        print(f"  skipped {n}: {reason}")
    if not any(r[2] == "valid" for r in rows):
        print("No validation images: train_jersey.py needs a valid/ (or val/) split")
    if rows and wide > len(rows) / 2:
        print(f"WARNING: {wide} of {len(rows)} images are wider than tall. Player crops are taller than wide; "
              "full broadcast frames won't work here, since the torso crop assumes one player per image.")
    mixed = [c for c, s in clips.items() if len(s) > 1]
    if len(clips) > 1 and mixed:
        print(f"WARNING: {len(mixed)} of {len(clips)} source clips (judged by file name) have images in more than "
              f"one split, e.g. {', '.join(sorted(mixed)[:3])}. Near-identical frames in train and valid make "
              "validation accuracy look better than it is; split by game in Roboflow if you can.")


if __name__ == "__main__":
    main()
