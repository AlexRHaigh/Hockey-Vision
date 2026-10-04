"""Train the jersey number ResNet (jersey_net.py) on labels.csv from make_jersey_labels.py.

    python colab/train_jersey.py path/to/dataset                  # whole-player crops, ResNet-34
    python colab/train_jersey.py path/to/dataset --torso          # images are already torso crops
    python colab/train_jersey.py path/to/dataset --arch resnet18  # about half the cost on the Jetson
    python colab/train_jersey.py path/to/dataset --arch resnet50 --input-hw 224 176 --batch 64
                                                    # the larger model: more capacity and detail

Train on Colab (train_jersey.ipynb), a Mac (MPS) or an NVIDIA PC, not on the Jetson. The best epoch, judged on the validation
split, is saved to CV_Models/jersey_Num Models/jersey_model.pt; copy that to the Jetson and build its engine with
export_engines.py --models jersey.

Validation reports three numbers:
    numbered  accuracy on crops that show a number (the one to watch)
    none      how often crops with no number are correctly read as none
    balanced  the mean of the two, which picks the best epoch, so a model can't score well just
              by answering "none" (or never answering it)
"""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# jersey_net.py is at the repo root (run_models.py uses it too); on Colab it sits next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jersey_net import ARCHS, INPUT_HW, MEAN, STD, JerseyNet, decode, encode, prepare, torso_crop

MODELS_DIR = Path(__file__).resolve().parent.parent / "CV_Models" / "jersey_Num Models"


class JerseyCrops(Dataset):
    def __init__(self, root, rows, transform, torso, input_hw):
        self.root, self.rows, self.transform, self.torso, self.input_hw = root, rows, transform, torso, input_hw

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img = cv2.imread(str(self.root / row["path"]))
        if img is None:
            raise OSError(f"Could not read {self.root / row['path']}")
        if not self.torso:  # a whole-player crop: cut the same band run_models.py will
            img = torso_crop(img, (0, 0, img.shape[1], img.shape[0]))
        x = torch.from_numpy(prepare(img, self.input_hw)).permute(2, 0, 1)  # uint8 RGB (3, H, W)
        return self.transform(x), torch.tensor(encode(row["number"]))


def low_res(x):
    """Downscale and back up again, like a far-away player in a broadcast frame."""
    h, w = x.shape[-2:]
    s = random.uniform(0.3, 0.7)
    return T.functional.resize(T.functional.resize(x, [max(8, int(h * s)), max(8, int(w * s))]), [h, w])


def transforms(train, input_hw):
    finish = [T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN.tolist(), STD.tolist())]
    if not train:
        return T.Compose(finish)
    # No horizontal flips: they mirror the digits.
    return T.Compose([
        T.RandomResizedCrop(input_hw, scale=(0.75, 1.0), ratio=(0.65, 1.0)),
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), shear=5),
        T.ColorJitter(0.4, 0.4, 0.4, 0.05),
        T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.3),
        T.RandomApply([T.Lambda(low_res)], p=0.3),
        T.RandomGrayscale(0.05),
        *finish,
    ])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    numbered = numbered_ok = none = none_ok = 0
    mistakes = Counter()
    for x, y in loader:
        preds = decode(*model(x.to(device)))
        for (pred, _), true in zip(preds, y[:, 0].tolist()):
            true = None if true == 100 else str(true)
            if true is None:
                none += 1
                none_ok += pred is None
            else:
                numbered += 1
                numbered_ok += pred == true
            if pred != true:
                mistakes[(true or "none", pred or "none")] += 1
    acc_num = numbered_ok / max(1, numbered)
    acc_none = none_ok / none if none else None
    balanced = acc_num if acc_none is None else (acc_num + acc_none) / 2
    return {"numbered": acc_num, "none": acc_none, "balanced": balanced}, mistakes


def fmt(metrics):
    return "  ".join(f"{k} {'-' if v is None else f'{v:.3f}'}" for k, v in metrics.items())


def main():
    parser = argparse.ArgumentParser(description="Train the jersey number ResNet.")
    parser.add_argument("dataset", type=Path, help="Folder containing labels.csv from make_jersey_labels.py")
    parser.add_argument("--torso", action="store_true", help="Images are already torso crops")
    parser.add_argument("--arch", choices=sorted(ARCHS), default="resnet34")
    parser.add_argument("--input-hw", type=int, nargs=2, default=list(INPUT_HW), metavar=("H", "W"),
                        help=f"Input height and width (default: {INPUT_HW[0]} {INPUT_HW[1]}; e.g. 224 176 for more "
                             "detail; keep it taller than wide, multiples of 16)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate (default: 1e-3)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "jersey_model.pt")
    args = parser.parse_args()

    input_hw = tuple(args.input_hw)
    rows = list(csv.DictReader(open(args.dataset / "labels.csv")))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "valid"]
    test_rows = [r for r in rows if r["split"] == "test"]
    if not train_rows or not val_rows:
        raise SystemExit("labels.csv needs both train and valid rows")
    print(f"{len(train_rows)} train, {len(val_rows)} valid, {len(test_rows)} test images")

    # Rare numbers are drawn more often (by 1/sqrt(count)), so the model doesn't only learn the
    # common ones, without repeating a handful of rare crops so often it memorises them.
    counts = Counter(r["number"] for r in train_rows)
    weights = [counts[r["number"]] ** -0.5 for r in train_rows]
    loader_kwargs = {"num_workers": args.workers, "persistent_workers": args.workers > 0}
    train_dl = DataLoader(JerseyCrops(args.dataset, train_rows, transforms(True, input_hw), args.torso, input_hw), batch_size=args.batch,
                          sampler=WeightedRandomSampler(weights, len(train_rows)), drop_last=True, **loader_kwargs)
    val_dl = DataLoader(JerseyCrops(args.dataset, val_rows, transforms(False, input_hw), args.torso, input_hw), batch_size=256,
                        **loader_kwargs)

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    model = JerseyNet(args.arch, input_hw=input_hw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr / 10, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    print(f"Training {args.arch} at {input_hw[0]}x{input_hw[1]} on {device} for {args.epochs} epochs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            number, tens, units = model(x)
            loss = ce(number, y[:, 0]) + 0.5 * (ce(tens, y[:, 1]) + ce(units, y[:, 2]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        metrics, _ = evaluate(model, val_dl, device)
        line = f"epoch {epoch:>3}/{args.epochs}  loss {total / len(train_dl):.3f}  val {fmt(metrics)}"
        if metrics["balanced"] > best:
            best = metrics["balanced"]
            torch.save({"state_dict": model.state_dict(), "arch": args.arch, "input_hw": list(input_hw),
                        "epoch": epoch, "val": metrics}, args.output)
            line += "  (saved)"
        print(line, flush=True)

    # Report the saved (best) epoch, with the mistakes it makes most.
    model.load_state_dict(torch.load(args.output, map_location=device)["state_dict"])
    for name, split_rows in (("valid", val_rows), ("test", test_rows)):
        if not split_rows:
            continue
        loader = DataLoader(JerseyCrops(args.dataset, split_rows, transforms(False, input_hw), args.torso, input_hw), batch_size=256,
                            num_workers=args.workers)
        metrics, mistakes = evaluate(model, loader, device)
        print(f"\nBest model on {name}: {fmt(metrics)}")
        print("  most common mistakes (true -> read): " +
              ", ".join(f"{t}->{p} x{n}" for (t, p), n in mistakes.most_common(10)))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
