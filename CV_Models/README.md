# CV_Models

Hockey-Vision runs five YOLO models we trained on broadcast NHL video. Together they find every
player, goalie, referee and the puck, read the players' jersey numbers, and find the rink markings
that `homography.py` uses to map each frame onto a top-down rink.

| Model | File | What it finds |
| --- | --- | --- |
| [Player](#player-model) | `new_player_model.pt` | Skaters and goalies for each team, and referees, tracked across frames |
| [Puck](#puck-model) | `new_nano_puck.pt` | The puck |
| [Number](#number-model) | `new_nums.pt` | Jersey digits, put together into each player's number (and name, with a roster) |
| [Rink](#rink-model) | `new_rink_model.pt` | Blue lines, centre line, circles, goal lines, goal posts and creases |
| [Dots](#dots-model) | `new_dots.pt` | The faceoff dots |

All five are YOLO26n detectors (about 2.5M parameters and 5 MB each), small enough to run
every frame on an NVIDIA Jetson Orin Nano.

[`Examples/`](Examples/) has each model running on the same three 10-second clips from
Canadiens at Sharks, March 3, 2026. San Jose is `team_a` and Montreal is `team_b`.

| Clip | Period, clock at the start | Score |
| --- | --- | --- |
| `clip1` | 1st, about 17:20 | MTL 0, SJS 0 |
| `clip2` | 1st, about 1:05 | MTL 1, SJS 1 |
| `clip3` | 3rd, about 19:00 | MTL 2, SJS 4 |

Each image below links to its 10-second video (1280×720, H.264). The example images and videos
aren't in git (`CV_Models/Examples/*/` is in `.gitignore`); to make them, see
[How the examples were made](#how-the-examples-were-made).

## Download

The weights aren't in git (`CV_Models/Models/` is in `.gitignore`). They are on Hugging Face at
[AlexRHaigh/Hockey-Vision](https://huggingface.co/AlexRHaigh/Hockey-Vision). From the repo root:

```bash
pip install -U huggingface_hub
hf download AlexRHaigh/Hockey-Vision --include "*.pt" --local-dir CV_Models/Models
```

```
CV_Models/
├── Models/                 the weights (downloaded, not in git)
│   ├── new_player_model.pt
│   ├── new_nano_puck.pt
│   ├── new_nums.pt
│   ├── new_rink_model.pt
│   └── new_dots.pt
└── Examples/               each model on three 10-second clips
    ├── player/  puck/  number/  rink/  dots/   clip1-3 .mp4 + .jpg (not in git)
    └── render_numbers.py   draws the number model's example videos
```

`run_models.py` reads the file names from `MODEL_FILES` and the folder from `MODELS_DIR`.

---

## Player model

**`new_player_model.pt`** · `run_models.py` name `player`

Finds every person on the ice and says who they play for:

| Class | Meaning |
| --- | --- |
| `team_a_player`, `team_b_player` | Skaters on each team |
| `goalie_a`, `goalie_b` | Each team's goalie |
| `referee` | Referees and linesmen |

The model also has a `puck` class, which `run_models.py` ignores because the puck model is
better at it. In videos the player boxes are tracked with ByteTrack, so each player keeps the same
`track_id` from frame to frame (the `id:` in the labels). That track id is what the number model's
votes and `export_data.py`'s distance skated are attached to.

## Puck model

**`new_nano_puck.pt`** · `run_models.py` name `puck`

Finds the puck. The model's one class is called `item`; `run_models.py` renames it to `puck`.
Below a confidence of about 0.6 its detections are often ad lettering, skates or gloves, so it runs
at `--puck-conf 0.6` by default, which is higher than the other models' 0.25.

## Number model

**`new_nums.pt`** · `run_models.py` name `number`

Reads jersey numbers. It's a digit detector (classes `0` to `9`, trained at 1280 px) that runs on
each player's box after the player model has found it. It works in four steps:

1. **Find the digits.** Each player crop is upscaled and the model finds single digits on it
   (the yellow boxes in the examples). Digits outside the torso, such as sock stripes, sticks or
   board ads, are thrown away. Referees aren't read.
2. **Put them together.** Starting from the most confident digit, at most one neighbouring digit
   of the same height and level with it is added, giving one- or two-digit numbers (`6`, `63`). A
   leading `0` is rejected because the NHL doesn't allow it.
3. **Vote across frames.** Each frame's reading is a confidence-weighted vote for that player's
   track. Once a number wins, it stays on the player in frames where the back of the jersey can't
   be seen, and one-off misreads are outvoted. A player whose number is settled is only re-read
   every `--jersey-stride` frames (default 5) to save time.
4. **Match to the roster** (with `--teams`). Numbers are limited to the ones the player's team
   wears, and each player is named from the roster CSV (`fetch_roster.py`).

In the examples each player is **marked with the number found on them and their name**
(`#63 Z. Ostapchuk`), in their team's colour: teal for San Jose, red for Montreal. Players stay
unmarked until enough votes have been collected for their number.


## Rink model

**`new_rink_model.pt`** · `run_models.py` name `rink`

Finds the painted rink markings:

| Class | Marking |
| --- | --- |
| `Blue_Line` | The two blue lines |
| `Center_Line` | The red centre line |
| `Center_Circle` | The centre-ice circle |
| `Circle` | The four end-zone faceoff circles |
| `Goal_Back_Line` | The goal lines |
| `Goal_Posts` | The goal frame |
| `Goal_Zone` | The goal crease |

On its own this model shows which part of the rink is in view. `homography.py` matches these boxes
(the ends of lines, the centres and edges of circles, the goal) to landmarks in `rink.py` with
known positions in feet. With the faceoff dots, that gives enough points to fit a homography from
each frame's pixels to rink coordinates. About 5 markings are found per frame in the examples.

## Dots model

**`new_dots.pt`** · `run_models.py` name `dots`

Finds the faceoff dots: the centre dot, the four neutral-zone dots and the four end-zone dots.
Dots are small, so the centre of each box is a precise point on the ice. `homography.py` trusts
them most when it fits the homography, and works out which dot is which from where it sits relative
to the lines and circles the rink model found. About 2 to 3 dots are in view per frame in the
examples.

---
