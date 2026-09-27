"""Build TensorRT engines from the models in CV_Models/, for running on the Jetson.

Run this on the Jetson itself: an engine only works on the GPU and TensorRT version it was built
with. Each engine is written next to its weights (CV_Models/<model>_model.engine), and
run_models.py uses it in place of the .pt from then on. Building takes a few minutes per model.

    python export_engines.py                 # all models, FP16
    python export_engines.py --models puck   # just one

Engines are built for a fixed input size, so run run_models.py with the same --imgsz (default
640). To go back to the .pt weights, delete the .engine files.
"""

import argparse

from ultralytics import YOLO

from run_models import MODEL_NAMES, MODELS_DIR, NUMBER_BATCH, NUMBER_IMGSZ


def main():
    parser = argparse.ArgumentParser(description="Export the Hockey-Vision models to TensorRT engines.")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES,
                        help="Which models to export (default: all)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size, must match run_models.py --imgsz")
    parser.add_argument("--fp32", action="store_true", help="Full precision instead of FP16 (slower)")
    args = parser.parse_args()

    for name in args.models:
        kwargs = {"format": "engine", "half": not args.fp32, "imgsz": args.imgsz, "device": 0}
        if name == "number":
            # The number model reads a batch of player crops per frame.
            kwargs.update(imgsz=NUMBER_IMGSZ, dynamic=True, batch=NUMBER_BATCH)
        print(f"Exporting {name}...")
        path = YOLO(str(MODELS_DIR / f"{name}_model.pt")).export(**kwargs)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
