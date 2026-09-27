"""Build TensorRT engines from the models in CV_Models/, for running on the Jetson.

Run this on the Jetson itself: an engine only works on the GPU and TensorRT version it was built
with. Each engine is written next to its weights (CV_Models/<model>_model.engine), and
run_models.py uses it in place of the .pt from then on. Building takes a few minutes per model.

    python export_engines.py                 # every model that doesn't have an engine yet, FP16
    python export_engines.py --models puck   # just one
    python export_engines.py --force         # rebuild engines that already exist

The Orin Nano's GPU shares the 8 GB of system RAM, and by default TensorRT grabs as much scratch
memory as it can while building, which fails with "NvMapMemAllocInternalTagged ... error 12".
--workspace caps it (1 GiB by default); lower it further if a build still runs out of memory.

Engines are built for a fixed input size, so run run_models.py with the same --imgsz (default
640). To go back to the .pt weights, delete the .engine files.
"""

import argparse
import subprocess
import sys

from ultralytics import YOLO

from run_models import MODEL_NAMES, MODELS_DIR, NUMBER_BATCH, NUMBER_IMGSZ


def main():
    parser = argparse.ArgumentParser(description="Export the Hockey-Vision models to TensorRT engines.")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES,
                        help="Which models to export (default: all)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size, must match run_models.py --imgsz")
    parser.add_argument("--workspace", type=float, default=1.0,
                        help="Max TensorRT build memory in GiB (default: 1; try 0.5 if a build runs out of memory)")
    parser.add_argument("--fp32", action="store_true", help="Full precision instead of FP16 (slower)")
    parser.add_argument("--force", action="store_true", help="Rebuild engines that already exist")
    args = parser.parse_args()

    if not args.force:
        for name in [n for n in args.models if (MODELS_DIR / f"{n}_model.engine").exists()]:
            print(f"Skipping {name}: {name}_model.engine already exists (--force to rebuild)")
        args.models = [n for n in args.models if not (MODELS_DIR / f"{n}_model.engine").exists()]

    if len(args.models) > 1:
        # One process per model, so each build starts with all of the Jetson's shared memory free.
        failed = []
        for name in args.models:
            cmd = [sys.executable, __file__, "--models", name, "--imgsz", str(args.imgsz),
                   "--workspace", str(args.workspace), "--force"] + (["--fp32"] if args.fp32 else [])
            if subprocess.run(cmd).returncode != 0:
                failed.append(name)
        if failed:
            sys.exit(f"Export failed for: {', '.join(failed)}. Retry with --models {' '.join(failed)}; "
                     "without an engine run_models.py uses the .pt for those models.")
        return

    for name in args.models:
        kwargs = {"format": "engine", "half": not args.fp32, "imgsz": args.imgsz, "device": 0,
                  "workspace": args.workspace}
        if name == "number":
            # The number model reads a batch of player crops per frame.
            kwargs.update(imgsz=NUMBER_IMGSZ, dynamic=True, batch=NUMBER_BATCH)
        print(f"Exporting {name}...")
        path = YOLO(str(MODELS_DIR / f"{name}_model.pt")).export(**kwargs)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
