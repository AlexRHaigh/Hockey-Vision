"""Build TensorRT engines from the models in CV_Models/Models/, for running on the Jetson.

Run this on the Jetson itself: an engine only works on the GPU and TensorRT version it was built
with. Each engine is written next to its weights in CV_Models/Models/ (new_player_model.engine /
new_nano_puck.engine / new_dots.engine / new_rink_model.engine / new_nums.engine for the player /
puck / dots / rink / number models), and run_models.py uses it in place of the .pt from then on. Building takes a few minutes per model.

    python export_engines.py                 # every model that doesn't have an engine yet, FP16
    python export_engines.py --models puck   # just one
    python export_engines.py --force         # rebuild engines that already exist
    python export_engines.py --int8 --calib-video videos/game.mp4   # INT8 engines, see below

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

INT8 (--int8) builds <model>.int8.engine next to the FP16 engine, which run_models.py --int8 uses
instead. INT8 needs calibration images that look like what the model sees at run time: frames
sampled evenly from --calib-video (for the player, rink and dots models) and the player crops in
them (for the number model), written once to CV_Models/Models/int8_calibration/. The puck model is
never built INT8: the puck is a few pixels across and its confidence threshold is tight, so it
stays FP16. Compare the INT8 detections with the FP16 ones before relying on them: a local test
of this calibration changed the number model's readings far more than the other models' boxes, so
`--models player rink dots` (number model left FP16) is the safer start. On JetPack 6's TensorRT
10.3, Ultralytics builds INT8 engines
without the NMS-free head (a TensorRT bug), so they run NMS after the model instead.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

from run_models import (FULL_FRAME_MODELS, INT8_MODELS, MIN_NUMBER_CROP_PX, MODEL_FILES, MODEL_NAMES,
                        MODELS_DIR, NUMBER_BATCH, NUMBER_IMGSZ, UNNUMBERED_CLASSES)

CALIB_DIR = MODELS_DIR / "int8_calibration"
CALIB_FRAMES = 500  # TensorRT's calibration wants a few hundred images


def quantize_kwargs(bits):
    """export() arguments for FP16 or INT8. Newer Ultralytics replaced `half` / `int8` with `quantize`."""
    if "quantize" in DEFAULT_CFG_DICT:
        return {"quantize": bits}
    return {"half": True} if bits == 16 else {"int8": True}


def build_calibration(video, n_frames=CALIB_FRAMES, out_dir=CALIB_DIR, device=None):
    """Write calibration images from `video` (frames, and the player crops in them) and return
    {"frames": dataset yaml, "crops": dataset yaml}. Reuses images already written."""
    import cv2

    frames_dir, crops_dir = out_dir / "frames" / "images", out_dir / "crops" / "images"
    if not (frames_dir.exists() and any(frames_dir.iterdir()) and crops_dir.exists() and any(crops_dir.iterdir())):
        if video is None:
            sys.exit("INT8 needs calibration images: pass --calib-video (a video like the ones you'll run on)")
        print(f"Writing INT8 calibration images from {video} to {out_dir}/")
        frames_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)
        player = YOLO(str(MODELS_DIR / f"{MODEL_FILES['player']}.pt"))
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            sys.exit(f"Could not read {video}")
        crops = 0
        for k in range(n_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * total / n_frames))
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(frames_dir / f"{k:04d}.jpg"), frame)
            r = player.predict(frame, conf=0.25, device=device, verbose=False)[0].cpu()
            for j, (box, cls) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())):
                x1, y1, x2, y2 = (max(0, int(v)) for v in box)
                if r.names[int(cls)] in UNNUMBERED_CLASSES | {"puck"} or y2 - y1 < MIN_NUMBER_CROP_PX:
                    continue
                cv2.imwrite(str(crops_dir / f"{k:04d}_{j:02d}.jpg"), frame[y1:y2, x1:x2])
                crops += 1
        cap.release()
        print(f"  {len(list(frames_dir.iterdir()))} frames, {crops} player crops")
    yamls = {}
    for kind in ("frames", "crops"):
        # No labels: calibration only needs the images. `names` just has to be a valid class list.
        path = out_dir / f"{kind}.yaml"
        path.write_text(f"path: {(out_dir / kind).resolve()}\ntrain: images\nval: images\nnames:\n  0: object\n")
        yamls[kind] = path
    return yamls


def main():
    parser = argparse.ArgumentParser(description="Export the Hockey-Vision models to TensorRT engines.")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES,
                        help="Which models to export (default: player puck number rink dots)")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[384, 640],
                        help="Input height and width (default: 384 640, for 16:9 video), or one value for square")
    parser.add_argument("--workspace", type=float, default=1.0,
                        help="Max TensorRT build memory in GiB (default: 1; try 0.5 if a build runs out of memory)")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--fp32", action="store_true", help="Full precision instead of FP16 (slower)")
    precision.add_argument("--int8", action="store_true",
                           help="Build <model>.int8.engine (INT8, calibrated on --calib-video) instead; "
                                "never for the puck model")
    parser.add_argument("--calib-video", type=Path, default=None,
                        help="With --int8, the video to take calibration frames from (once; they're kept in "
                             "CV_Models/Models/int8_calibration/)")
    parser.add_argument("--force", action="store_true", help="Rebuild engines that already exist")
    args = parser.parse_args()

    suffix = ".int8.engine" if args.int8 else ".engine"
    if args.int8:
        for name in [n for n in args.models if n not in INT8_MODELS]:
            print(f"Skipping {name}: it always runs FP16 (see INT8_MODELS in run_models.py)")
        args.models = [n for n in args.models if n in INT8_MODELS]
    if not args.force:
        for name in [n for n in args.models if (MODELS_DIR / f"{MODEL_FILES[n]}{suffix}").exists()]:
            print(f"Skipping {name}: {MODEL_FILES[name]}{suffix} already exists (--force to rebuild)")
        args.models = [n for n in args.models if not (MODELS_DIR / f"{MODEL_FILES[n]}{suffix}").exists()]

    if args.int8 and args.models:
        calib = build_calibration(args.calib_video, device=0)  # once, before the per-model processes

    if len(args.models) > 1:
        # One process per model, so each build starts with all of the Jetson's shared memory free.
        failed = []
        for name in args.models:
            cmd = [sys.executable, __file__, "--models", name, "--imgsz", *map(str, args.imgsz),
                   "--workspace", str(args.workspace), "--force"] + (["--fp32"] if args.fp32 else [])
            if args.int8:
                cmd += ["--int8"]
            if subprocess.run(cmd).returncode != 0:
                failed.append(name)
        if failed:
            sys.exit(f"Export failed for: {', '.join(failed)}. Retry with --models {' '.join(failed)}; "
                     "without an engine run_models.py uses the .pt for those models.")
        return

    for name in args.models:
        imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz[:2]
        kwargs = {"format": "engine", "imgsz": imgsz, "device": 0, "workspace": args.workspace}
        if args.int8:
            kwargs.update(quantize_kwargs(8), data=str(calib["frames" if name in FULL_FRAME_MODELS else "crops"]))
        elif not args.fp32:
            kwargs.update(quantize_kwargs(16))
        if name == "number":
            # The number model reads a batch of player crops per frame.
            kwargs.update(imgsz=NUMBER_IMGSZ, dynamic=True, batch=NUMBER_BATCH)
        weights = MODELS_DIR / f"{MODEL_FILES[name]}.pt"
        if args.int8:
            # Export from a copy, so the INT8 build doesn't overwrite the FP16 <model>.engine.
            build_dir = CALIB_DIR / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            weights = Path(shutil.copy(weights, build_dir / weights.name))
        print(f"Exporting {name}{' (INT8)' if args.int8 else ''}...")
        path = Path(YOLO(str(weights)).export(**kwargs))
        if args.int8:
            path = Path(shutil.move(str(path), str(MODELS_DIR / f"{MODEL_FILES[name]}{suffix}")))
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
