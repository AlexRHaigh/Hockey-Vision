# Hockey-Vision

Hockey-Vision turns broadcast NHL video into player and puck tracking data and play-by-play
events. It finds every player, referee and the puck in each frame, reads the players' jersey
numbers, works out where each frame's camera is looking on the rink, and maps everyone onto a
top-down rink in real-world feet. From those rink positions it finds possessions, shots, passes,
turnovers and defensive plays. The output is machine-readable data: per-frame and per-player
tables, an event list with times, and a `game_report.json` written for a language model to read
and turn into feedback.

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
export_data.py ── tables + play events (events.py) ──────► outputs/<clip>/export/
                   possessions, shots, passes, turnovers,    *.csv, game_report.json
                   defensive plays, per-player stats
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
3. **Export and play events** (`export_data.py`, `events.py`): one or more clips' outputs become
   tables, play events and per-player stats (see [Output](#output)). Events are found from the
   rink positions alone, in the 2D rink plane.

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
   (`positions.json`, `positions.csv`, `homographies.csv`).
3. `export_data.py outputs/<clip>` writes the tables, events and `game_report.json` to
   `outputs/<clip>/export/` (several clips: `export_data.py outputs`, written to `outputs/export/`).

`run_models.py --width <px>` (e.g. 640, 960, 1280) resizes the frames before the models see them,
to test how they do on lower-resolution video; the default is the video's own resolution.
Detections are still written in the source video's pixels, so the later steps run unchanged.

`jetson/run_clip.sh <video> [--teams SJS MTL]` runs all three. Only data is written by default.
Annotated videos are optional outputs, for checking the models by eye:
`run_models.py --save-video` writes `combined.mp4` (`--per-model` adds one video per model), and
`homography.py --save-video` writes `side_by_side.mp4` with the top-down rink beside the video
(`--debug` adds `radar.mp4` and `overlay.mp4`).

## Output

`outputs/<clip>/export/`:

| File | Contents |
|---|---|
| `game_report.json` | Everything a language model needs for feedback, in one file: definitions, team totals, each player's stats, and every event with its time and a plain-English `description` |
| `events.csv` | One row per event: `type` (`shot`, `pass`, `turnover`, `defensive_play`), `subtype`, time, who did it, who to/from, rink position, zone; type-specific fields in `details` (JSON) |
| `player_stats.csv` | One row per player: time detected, distance skated, possessions, shots, passes (good ones too), takeaways, interceptions, turnovers, defensive plays |
| `possessions.csv` | One row per spell of a player carrying the puck, and how it ended |
| `players.csv`, `puck.csv`, `frames.csv`, `tracks.csv` | Per-frame positions and per-track summaries ([jetson/README.md](jetson/README.md#exported-tables)) |
| `metadata.json` | Source files, frame rate, coordinate system and a description of every column |

The events (`events.py` has the exact rules and thresholds):

- **Shots**: the puck leaves a player's stick at shot speed on a line at the other team's net, or
  the other team's goalie gets it straight after a release near the net. Outcomes: `saved`,
  `blocked`, `possible_goal`, `rebound_recovered`, `recovered_by_opponent`, `unknown`.
- **Passes**: the puck goes from one player to a teammate. `good` passes say why: advanced the
  puck, beat defenders, relieved pressure, found an open teammate, or led to a shot.
- **Turnovers**: the other team gets the puck: a `takeaway` (stolen off the carrier), an
  `interception`, or a `goalie_recovery`.
- **Defensive plays**: an opponent close to the carrier forces them to backtrack, slows them
  down, or makes them pass the puck away. Credited to that defender.
- **Time detected**: per player, every frame of every track carrying their jersey number.

Which way each team attacks comes from where the goalies stand. Times are video time, not the
game clock. The puck is only seen in part of the frames and players are only named once their
number is read, so treat the counts as estimates.

## Repository layout

| Path | Contents |
|---|---|
| `run_models.py` | Runs the detection models, tracks players and reads jersey numbers |
| `homography.py`, `rink.py` | Maps detections onto the rink; rink geometry and the template drawing |
| `export_data.py`, `events.py` | Turns a clip's outputs into tables, play events and `game_report.json` |
| `fetch_roster.py`, `roster.py`, `rosters/` | NHL rosters for `--teams` / `--roster` |
| `export_engines.py`, `jetson/` | TensorRT engines and the Docker setup for the Jetson |
| `video_io.py` | Background-thread video decoding |
| `CV_Models/` | The models (see [CV_Models/README.md](CV_Models/README.md)) |
| `assets/` | The rink template image (for the optional videos) |
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
