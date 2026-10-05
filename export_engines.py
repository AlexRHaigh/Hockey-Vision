"""Build TensorRT engines from the models in CV_Models/Models/, for running on the Jetson.

Run this on the Jetson itself: an engine only works on the GPU and TensorRT version it was built
with. Each engine is written next to its weights in CV_Models/Models/ (new_player_model.engine /
new_nano_puck.engine / new_dots.engine / new_rink_model.engine / new_nums.engine for the player /
puck / dots / rink / number models), and run_models.py uses it in place of the .pt from then on. Building takes a few minutes per model.

    python export_engines.py                 # every model that doesn't have an engine yet, FP16
    python export_engines.py --models puck   # just one
    python export_engines.py --force         # rebuild engines that already exist

The Orin Nano's GPU shares the 8 GB of system RAM, and by default TensorRT grabs as much scratch
memory as it can while building, which fails with "NvMapMemAllocInternalTagged ... error 12".
--workspace caps it (1 GiB by default); lower it further if a build still runs out of memory.

Engines are built for a fixed input size, 384x640 (height x width) by default: a 16:9 frame
scaled to 640 wide, which is what the .pt weights see at run_models.py's default --imgsz 640. A
square 640x640 engine would spend ~40% of its time on the letterbox padding above and below the
frame. For video of another shape pass --imgsz H W (multiples of 32), or one value for a square
engine. run_models.py reads each engine's size from the engine itself. Engines built before this
default were square: rebuild them with --force. To go back to the .pt weights, delete the
.engine files.

Jersey numbers are read by the YOLO number model (new_nums.pt), whose engine is built by default
for batches of up to NUMBER_BATCH player crops.
"""

import argparse
import subprocess
import sys

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

from run_models import MODEL_NAMES, MODELS_DIR, NUMBER_BATCH, NUMBER_IMGSZ, weights_stem


def main():
    parser = argparse.ArgumentParser(description="Export the Hockey-Vision models to TensorRT engines.")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES,
                        help="Which models to export (default: player puck number rink dots)")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[384, 640],
                        help="Input height and width (default: 384 640, for 16:9 video), or one value for square")
    parser.add_argument("--workspace", type=float, default=1.0,
                        help="Max TensorRT build memory in GiB (default: 1; try 0.5 if a build runs out of memory)")
    parser.add_argument("--fp32", action="store_true", help="Full precision instead of FP16 (slower)")
    parser.add_argument("--force", action="store_true", help="Rebuild engines that already exist")
    args = parser.parse_args()

    if not args.force:
        for name in [n for n in args.models if (MODELS_DIR / f"{weights_stem(n)}.engine").exists()]:
            print(f"Skipping {name}: {weights_stem(name)}.engine already exists (--force to rebuild)")
        args.models = [n for n in args.models if not (MODELS_DIR / f"{weights_stem(n)}.engine").exists()]

    if len(args.models) > 1:
        # One process per model, so each build starts with all of the Jetson's shared memory free.
        failed = []
        for name in args.models:
            cmd = [sys.executable, __file__, "--models", name, "--imgsz", *map(str, args.imgsz),
                   "--workspace", str(args.workspace), "--force"] + (["--fp32"] if args.fp32 else [])
            if subprocess.run(cmd).returncode != 0:
                failed.append(name)
        if failed:
            sys.exit(f"Export failed for: {', '.join(failed)}. Retry with --models {' '.join(failed)}; "
                     "without an engine run_models.py uses the .pt for those models.")
        return

    for name in args.models:
        imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz[:2]
        kwargs = {"format": "engine", "imgsz": imgsz, "device": 0, "workspace": args.workspace}
        if not args.fp32:  # newer Ultralytics replaced `half` with `quantize`
            kwargs.update({"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True})
        if name == "number":
            # The number model reads a batch of player crops per frame.
            kwargs.update(imgsz=NUMBER_IMGSZ, dynamic=True, batch=NUMBER_BATCH)
        print(f"Exporting {name}...")
        path = YOLO(str(MODELS_DIR / f"{weights_stem(name)}.pt")).export(**kwargs)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
