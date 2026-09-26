"""Run the Hockey-Vision YOLO models on an image, a video, or a folder of either.

Examples:
    python run_models.py path/to/frame.jpg
    python run_models.py path/to/game.mp4 --models player puck --conf 0.3
    python run_models.py path/to/folder --no-per-model --output results

For every input this writes to <output>/<input name>/:
    combined.<ext>        all selected models drawn on one image/video
    <model>.<ext>         one annotated output per model (unless --no-per-model)
    detections.json       every detection (per frame for videos)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

MODELS_DIR = Path(__file__).parent / "CV_Models"
MODEL_NAMES = ["player", "puck", "number", "rink", "dots"]

# BGR colour per model for the combined output, so each model's boxes are distinguishable.
MODEL_COLORS = {
    "player": (60, 180, 75),
    "puck": (0, 0, 255),
    "number": (0, 215, 255),
    "rink": (255, 130, 0),
    "dots": (240, 50, 230),
}

# Classes to drop from a model's output. The puck model handles pucks, so the player model's
# puck class is ignored to avoid duplicate boxes.
EXCLUDED_CLASSES = {
    "player": {"puck"},
}

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


def load_models(names):
    models = {}
    for name in names:
        path = MODELS_DIR / f"{name}_model.pt"
        if not path.exists():
            sys.exit(f"Model not found: {path}")
        models[name] = YOLO(str(path))
    return models


def kept_classes(name, model):
    """Class ids to keep for a model, or None to keep all of them."""
    excluded = EXCLUDED_CLASSES.get(name)
    if not excluded:
        return None
    return [i for i, cls in model.names.items() if cls not in excluded]


def run_on_frame(models, frame, conf, imgsz, device):
    """Return {model_name: ultralytics Results} for a single BGR frame."""
    return {
        name: model.predict(frame, conf=conf, imgsz=imgsz, device=device,
                            classes=kept_classes(name, model), verbose=False)[0]
        for name, model in models.items()
    }


def results_to_dicts(results):
    out = {}
    for name, r in results.items():
        dets = []
        for box, cls, score in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            dets.append({
                "class": r.names[int(cls)],
                "confidence": round(score, 4),
                "box_xyxy": [round(v, 1) for v in box],
            })
        out[name] = dets
    return out


def draw_label(img, text, x, y, color):
    scale = max(0.4, img.shape[1] / 2500)
    thickness = max(1, int(scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y = max(y, th + baseline)
    cv2.rectangle(img, (x, y - th - baseline), (x + tw, y), color, -1)
    cv2.putText(img, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def draw_combined(frame, results):
    img = frame.copy()
    line = max(1, int(img.shape[1] / 800))
    for name, r in results.items():
        color = MODEL_COLORS.get(name, (200, 200, 200))
        for box, cls, score in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line)
            draw_label(img, f"{r.names[int(cls)]} {score:.2f}", x1, y1, color)

    # Legend in the top-left corner.
    y = 10
    for name in results:
        y += 25
        draw_label(img, name, 10, y, MODEL_COLORS.get(name, (200, 200, 200)))
    return img


def process_image(path, models, args, out_dir):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"  Could not read {path}, skipping")
        return
    results = run_on_frame(models, frame, args.conf, args.imgsz, args.device)

    cv2.imwrite(str(out_dir / f"combined{path.suffix}"), draw_combined(frame, results))
    if args.per_model:
        for name, r in results.items():
            cv2.imwrite(str(out_dir / f"{name}{path.suffix}"), r.plot())

    detections = results_to_dicts(results)
    (out_dir / "detections.json").write_text(json.dumps(detections, indent=2))
    counts = ", ".join(f"{n}: {len(d)}" for n, d in detections.items())
    print(f"  {counts}")


def process_video(path, models, args, out_dir):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  Could not open {path}, skipping")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {"combined": cv2.VideoWriter(str(out_dir / "combined.mp4"), fourcc, fps, (w, h))}
    if args.per_model:
        for name in models:
            writers[name] = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), fourcc, fps, (w, h))

    all_detections = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and idx >= args.max_frames):
                break
            results = run_on_frame(models, frame, args.conf, args.imgsz, args.device)

            writers["combined"].write(draw_combined(frame, results))
            if args.per_model:
                for name, r in results.items():
                    writers[name].write(r.plot())

            all_detections.append({"frame": idx, "time_s": round(idx / fps, 3),
                                   "detections": results_to_dicts(results)})
            idx += 1
            if idx % 25 == 0 or idx == total:
                print(f"\r  frame {idx}/{total or '?'}", end="", flush=True)
    finally:
        cap.release()
        for wr in writers.values():
            wr.release()
    print()

    (out_dir / "detections.json").write_text(json.dumps(all_detections, indent=2))


def collect_inputs(source):
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
    return [source]


def main():
    parser = argparse.ArgumentParser(description="Run the Hockey-Vision models and save annotated outputs.")
    parser.add_argument("source", type=Path, help="Image, video, or folder of images/videos")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES,
                        help="Which models to run (default: all)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--device", default=None, help="cpu, mps, cuda, 0... (default: auto)")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output folder (default: outputs)")
    parser.add_argument("--no-per-model", dest="per_model", action="store_false",
                        help="Only write the combined output, not one per model")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop videos after N frames (0 = all)")
    args = parser.parse_args()

    if not args.source.exists():
        sys.exit(f"Source not found: {args.source}")
    args.device = pick_device(args.device)

    inputs = collect_inputs(args.source)
    if not inputs:
        sys.exit(f"No images or videos found in {args.source}")

    print(f"Loading models: {', '.join(args.models)} (device: {args.device})")
    models = load_models(args.models)

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
