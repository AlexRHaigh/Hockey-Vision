"""Turn a YOLO digit dataset (one box per digit, like the one number_model.pt was trained on) into
whole-number labels for train_jersey.py.

    python make_jersey_labels.py path/to/dataset            # images are whole-player crops
    python make_jersey_labels.py path/to/dataset --torso    # images are already torso crops

The dataset folder needs data.yaml (class names "0".."9") and images with YOLO .txt labels, in
either layout: <split>/images + <split>/labels (Roboflow) or images/<split> + labels/<split>
(Ultralytics). It writes <dataset>/labels.csv with columns path, number, split; number is empty
for an image with no digits, which teaches the model to say "no readable number".

The digits in an image are joined left to right into the number. Images with more than two
digits (a sleeve number as well as the back one) or a leading zero are skipped. For whole-player
crops, images whose digits fall outside the torso band the model sees (jersey_net.TORSO_Y) are
skipped too, since the number wouldn't be in its crop.

Splits come from folder names (train / valid / val / test). Keep each game in one split: frames
of the same player in both train and validation make validation accuracy meaningless.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

from jersey_net import TORSO_Y

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def digit_names(data_yaml):
    names = yaml.safe_load(data_yaml.read_text())["names"]
    names = names if isinstance(names, list) else [names[i] for i in sorted(names)]
    digits = []
    for n in names:
        d = [c for c in str(n) if c.isdigit()]
        if len(d) != 1:
            sys.exit(f"Class {n!r} in {data_yaml} isn't a single digit; this script expects classes 0-9")
        digits.append(d[0])
    return digits


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

    names = digit_names(args.dataset / "data.yaml")
    images = sorted(p for p in args.dataset.rglob("*") if p.suffix.lower() in IMAGE_EXTS and "images" in p.parts)
    if not images:
        sys.exit(f"No images under an images/ folder in {args.dataset}")

    skipped = Counter()
    rows = []
    for img in images:
        split = next((SPLITS[p] for p in img.relative_to(args.dataset).parts if p in SPLITS), None)
        if split is None:
            skipped["no train/valid/test folder in its path"] += 1
            continue
        lbl = label_path(img)
        boxes = [line.split() for line in lbl.read_text().splitlines() if line.strip()] if lbl.exists() else []
        # YOLO rows: class x_centre y_centre width height, all 0-1.
        digits = sorted((float(b[1]), float(b[2]), names[int(b[0])]) for b in boxes)
        if len(digits) > 2:
            skipped["more than two digits"] += 1
            continue
        if not args.torso and any(not TORSO_Y[0] <= y <= TORSO_Y[1] for _, y, _ in digits):
            skipped["digits outside the torso crop"] += 1
            continue
        number = "".join(d for _, _, d in digits)
        if number.startswith("0"):
            skipped["leading zero"] += 1
            continue
        rows.append([img.relative_to(args.dataset).as_posix(), number, split])

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


if __name__ == "__main__":
    main()
