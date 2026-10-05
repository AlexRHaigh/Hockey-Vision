# Hockey-Vision

Hockey-Vision turns broadcast NHL video into player and puck tracking data. It finds every
player, referee and the puck in each frame, reads the players' jersey numbers, works out where
each frame's camera is looking on the rink, and maps everyone onto a top-down rink in real-world
feet. The output is a set of per-frame and per-player tables (positions, teams, numbers, names,
distance skated) that can be analysed like tracking data.

It runs on a Mac or a CUDA GPU, and on an NVIDIA Jetson Orin Nano with TensorRT engines (see
[jetson/README.md](jetson/README.md)).

## Design

```
broadcast video
      │
      ▼
run_models.py ── five YOLO models per frame ──────────────► outputs/<clip>/detections.json
      │   player  players, goalies (team a / b), referees; tracked across frames (ByteTrack)
      │   puck    the puck
      │   number  jersey digits on each player's box, voted per track → jersey number
      │   rink    rink keypoints (lines, circles, goals)
      │   dots    the faceoff dots on the ice
      │   (+ optional --teams roster: limits numbers to each team's and names the players)
      ▼
homography.py ── rink + dots keypoints → per-frame homography → rink feet ─► positions.json/.csv
      │   matched to named landmarks in rink.py, outliers dropped, smoothed over time
      ▼
export_data.py ── analysis tables ───────────────────────► outputs/<clip>/export/*.csv
```

1. **Detection and tracking** (`run_models.py`): the player model's boxes are tracked across
   frames, so each player keeps an id. The number model reads digits on each player's torso,
   and every frame's reading votes for that track's number, so the number holds through frames
   where it can't be read. With `--teams`, a roster from `fetch_roster.py` restricts the numbers
   to those each team wears and names the players (`roster.py`).
2. **Rink mapping** (`homography.py`, `rink.py`): the rink and dots models' keypoints are matched
   to known rink landmarks, and a homography from image pixels to rink feet is fitted for each
   frame. Players are placed where their skates meet the ice. Rink coordinates are feet from the
   centre dot: x along the rink (-100 to 100), y across it (-42.5 to 42.5).
3. **Export** (`export_data.py`): one or more clips' outputs become `players.csv`, `puck.csv`,
   `frames.csv`, `tracks.csv` and `metadata.json` (described in
   [jetson/README.md](jetson/README.md#exported-tables)).

## Models

The five YOLO models aren't in git; they're on Hugging Face at
[AlexRHaigh/Hockey-Vision](https://huggingface.co/AlexRHaigh/Hockey-Vision) and go in
`CV_Models/Models/`. **[CV_Models/README.md](CV_Models/README.md) lists each model, what it
detects and how to download them.**

## Setup

```bash
pip install -r requirements.txt
pip install -U huggingface_hub
hf download AlexRHaigh/Hockey-Vision --include "*.pt" --local-dir CV_Models/Models
```

## Running

1. `run_models.py <video>` runs the detection models and writes `outputs/<clip>/detections.json`.
2. `homography.py outputs/<clip>/detections.json <video>` maps players and the puck onto the rink
   (`positions.json`, `positions.csv`, `homographies.csv`, and `side_by_side.mp4`).
3. `export_data.py outputs/<clip>` writes analysis-ready tables to `outputs/<clip>/export/`:
   `players.csv`, `puck.csv`, `frames.csv`, `tracks.csv` and `metadata.json`
   (see [jetson/README.md](jetson/README.md#exported-tables)).

## Repository layout

| Path | Contents |
|---|---|
| `run_models.py` | Runs the detection models, tracks players and reads jersey numbers |
| `homography.py`, `rink.py` | Maps detections onto the rink; rink geometry and the template drawing |
| `export_data.py` | Turns a clip's outputs into analysis tables |
| `fetch_roster.py`, `roster.py`, `rosters/` | NHL rosters for `--teams` / `--roster` |
| `export_engines.py`, `jetson/` | TensorRT engines and the Docker setup for the Jetson |
| `video_io.py` | Background-thread video decoding |
| `CV_Models/` | The models (see [CV_Models/README.md](CV_Models/README.md)) |
| `assets/` | The rink template image |
| `tests/` | Tests |

## Jersey numbers

`run_models.py` reads jersey numbers with the YOLO number model, `CV_Models/Models/new_nums.pt` (YOLO26n,
classes `0`-`9`). It detects single digits on each player's box and joins them into the number;
digits outside the torso band (stripes on socks, sticks, the boards) are ignored. Each player
track's readings are combined in a vote, so a number stays with the player through frames where
it can't be read and one-off misreads are outvoted. Referees aren't read.

### Rosters and player names

Give the two teams' abbreviations, team_a's first (the player model's `team_a_player` / `goalie_a`
classes), and jersey numbers are limited to what each team wears, players are named, and referees
get no number:

```bash
python fetch_roster.py                                   # all 32 teams' active rosters -> rosters/nhl_active_players.csv
python run_models.py videos/<clip>.mp4 --teams SJS MTL   # team_a = SJS, team_b = MTL
python fetch_roster.py --game 2025020969                 # a past game: exactly who dressed
python run_models.py videos/<clip>.mp4 --teams SJS MTL --roster rosters/2026-03-03_MTL_at_SJS.csv
```

A reading that isn't a number that team wears is then dropped. `detections.json`, `positions.csv`
and the exported `players.csv` / `tracks.csv` gain `team_abbrev` / `player_name`. The roster CSV has one row per player:
`team, team_name, number, player, position, player_id, as_of`; refetch it as rosters change.

`detections.json` lists the digit boxes in `number`.
