# Running on a Jetson Orin Nano

Setup: JetPack 6 on the Orin Nano, everything run inside Ultralytics' Jetson Docker image
(CUDA PyTorch from `pip` doesn't exist for the Jetson, so a plain `pip install -r requirements.txt`
would run on the CPU).

## 1. One-time setup on the Jetson

```bash
sudo nvpmodel -q                        # current power mode; switch to the highest (MAXN / MAXN SUPER)
sudo nvpmodel -m <mode>                 #   with -m, the mode numbers are listed in /etc/nvpmodel.conf
sudo jetson_clocks                      # run the clocks at max (resets on reboot)
git clone https://github.com/AlexRHaigh/Hockey-Vision.git
cd Hockey-Vision
sudo docker build -t hockey-vision jetson/
```

Put the image and the outputs on the NVMe drive if the Jetson has one; the image is several GB.

## 2. Copy the models and videos over (from your Mac)

`CV_Models/` and `videos/` aren't in git.

```bash
scp -r CV_Models videos <user>@<jetson-ip>:~/Hockey-Vision/
```

## 3. Start the container

```bash
cd ~/Hockey-Vision
sudo docker run -it --rm --runtime=nvidia --ipc=host -v "$PWD":/work hockey-vision bash
```

The repo is mounted at `/work`, so everything written there lands in `~/Hockey-Vision` on the
Jetson and survives the container.

## 4. Build the TensorRT engines (once, inside the container)

```bash
python export_engines.py
```

This writes `CV_Models/<model>_model.engine` for each model (`new_player_model.engine`, `new_nano_puck.engine`, `new_dots.engine` and `new_rink_model.engine` for the player, puck, dots and rink, and the number model's in `CV_Models/jersey_Num Models/`; FP16), which `run_models.py` then
uses automatically. It takes a while. The engines take 384x640 input (16:9 video at 640 wide), not
640x640, which saves the ~40% of the work a square engine spends on padding; for video of another
shape pass `--imgsz <height> <width>`. Engines built before this was the default are square:
rebuild them with `python export_engines.py --force`.

Jersey numbers are read by PARSeq (`CV_Models/jersey_Num Models/jersey.ckpt`) after a legibility check
(`CV_Models/jersey_Num Models/legibility_resnet34_hockey_20240201.pth`); copy both over with the other models. PARSeq
runs in PyTorch in FP16 and needs no engine; the Docker image installs its code (`timm` and
the PARSeq repo), so rebuild the image if it predates that. The YOLO `number_model` engine is still
built, for `--number-reader yolo`; skip it with `--models player puck rink dots`. If a build runs
out of memory, add swap on the Jetson and retry just that model with `--models <name>`.

## 5. Run a clip

```bash
jetson/run_clip.sh videos/<clip>.mp4                    # whole video
jetson/run_clip.sh videos/<clip>.mp4 --max-frames 1800  # first 1800 frames only
jetson/run_clip.sh videos/<clip>.mp4 --teams SJS MTL    # team_a, team_b: numbers from their rosters, players named
```

`--teams` reads the rosters from `rosters/nhl_active_players.csv` (make it with
`python fetch_roster.py`, which needs internet; copy it over if the Jetson is offline), or from a
past game's file with `--roster rosters/<date>_<away>_at_<home>.csv` (`python fetch_roster.py --game <id>`).

This skips all annotated videos and reads every player's jersey number with PARSeq on every
frame. It writes, in `outputs/<clip>/`:

| File | Contents |
|---|---|
| `homographies.csv` | one row per frame: `frame, time_s, rejected, keypoints_used, h00 … h22` (empty when no fit) |
| `positions.csv` | one row per player / puck per frame: `frame, time_s, object, track_id, class, jersey_number, confidence, x_ft, y_ft` |
| `positions.json` | the same, nested per frame |
| `detections.json` | the raw model detections (input to `homography.py`) |
| `export/` | analysis-ready tables from `export_data.py`, see below |

To also get `side_by_side.mp4`, run `homography.py` without `--no-video`.

## 6. Copy the results back (from your Mac)

```bash
scp <user>@<jetson-ip>:~/Hockey-Vision/outputs/<clip>/{homographies.csv,positions.csv,positions.json} .
```

or just the export tables: `scp -r <user>@<jetson-ip>:~/Hockey-Vision/outputs/<clip>/export .`

## Exported tables

`export_data.py` (the last step of `run_clip.sh`) joins the detections and rink positions into
tables with a `clip` column, so several clips can be exported together
(`python export_data.py outputs` writes every clip to `outputs/export/`). It only needs the
Python standard library, so it can also be re-run on your Mac on copied output folders.

| File | Contents |
|---|---|
| `players.csv` | one row per player detection per frame: `track_id`, `team` (A/B), `role` (skater/goalie/referee), `jersey_number`, image box, and rink `x_ft, y_ft` (empty on frames without a usable fit) |
| `puck.csv` | one row per frame with a puck: image box and rink `x_ft, y_ft` |
| `frames.csv` | one row per frame: `rink_fit`, `rejected`, `keypoints_used`, player and puck counts |
| `tracks.csv` | one row per player track: team, role, jersey number, first/last frame, frames seen, mean position, `distance_ft`, `mean_speed_ft_s` |
| `metadata.json` | source files, fps, the coordinate system and a description of every column |

## Using the homography

Each row of `homographies.csv` is the 3x3 matrix mapping a video pixel `(u, v)` to rink feet
(origin at centre ice, x -100..100 along the rink, y -42.5 near boards .. 42.5 far boards):

```python
import numpy as np, pandas as pd

H = pd.read_csv("homographies.csv").dropna(subset=["h00"])
row = H.iloc[0]
M = row[[f"h{r}{c}" for r in range(3) for c in range(3)]].to_numpy(float).reshape(3, 3)
x, y, w = M @ [960, 700, 1]
print(row.frame, x / w, y / w)   # rink feet of pixel (960, 700) in that frame
```
