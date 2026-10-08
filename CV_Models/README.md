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

`team_a` and `team_b` are learned from the jersey colours, not from team names. Pass `--teams` to
say which team is which (here `--teams SJS MTL`).

About 8 to 9 people are found in each frame of the example clips.

| clip1 | clip2 | clip3 |
| --- | --- | --- |
| [![player clip1](Examples/player/clip1.jpg)](Examples/player/clip1.mp4) | [![player clip2](Examples/player/clip2.jpg)](Examples/player/clip2.mp4) | [![player clip3](Examples/player/clip3.jpg)](Examples/player/clip3.mp4) |

## Puck model

**`new_nano_puck.pt`** · `run_models.py` name `puck`

Finds the puck. The model's one class is called `item`; `run_models.py` renames it to `puck`.
Below a confidence of about 0.6 its detections are often ad lettering, skates or gloves, so it runs
at `--puck-conf 0.6` by default, which is higher than the other models' 0.25.

The puck is small and often hidden behind players or blurred, so it isn't found in every frame. In
the example clips it was found in 24%, 19% and 32% of frames. `homography.py` places it on the
rink in the frames where it is found.

| clip1 | clip2 | clip3 |
| --- | --- | --- |
| [![puck clip1](Examples/puck/clip1.jpg)](Examples/puck/clip1.mp4) | [![puck clip2](Examples/puck/clip2.jpg)](Examples/puck/clip2.mp4) | [![puck clip3](Examples/puck/clip3.jpg)](Examples/puck/clip3.mp4) |

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

| clip1 | clip2 | clip3 |
| --- | --- | --- |
| [![number clip1](Examples/number/clip1.jpg)](Examples/number/clip1.mp4) | [![number clip2](Examples/number/clip2.jpg)](Examples/number/clip2.mp4) | [![number clip3](Examples/number/clip3.jpg)](Examples/number/clip3.mp4) |

Players identified in each clip:

| Clip | San Jose (team_a) | Montreal (team_b) |
| --- | --- | --- |
| clip1 | #2 Will Smith, #3 John Klingberg, #5 Vincent Desharnais, #6 Sam Dickinson, #63 Zack Ostapchuk | #8 Mike Matheson, #11 Brendan Gallagher, #14 Nick Suzuki, #17 Josh Anderson, #21 Kaiden Guhle, #24 Phillip Danault |
| clip2 | #2 Will Smith, #3 John Klingberg, #9 Dmitry Orlov, #23 Barclay Goodrow, #30 Yaroslav Askarov (G), #63 Zack Ostapchuk, #81 Adam Gaudette | #8 Mike Matheson, #11 Brendan Gallagher, #14 Nick Suzuki |
| clip3 | #2 Will Smith, #3 John Klingberg, #5 Vincent Desharnais, #6 Sam Dickinson, #9 Dmitry Orlov, #63 Zack Ostapchuk, #81 Adam Gaudette, #85 Shakir Mukhamadullin | #8 Mike Matheson, #11 Brendan Gallagher, #20 Juraj Slafkovský, #76 Zachary Bolduc |

These are the model's readings, not checked against the broadcast frame by frame. A misread
can still win the vote: in the clip3 image the player near the goal is tagged `#8 M. Matheson`,
but the Matheson #8 jersey is on the untagged player at the bottom right.

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

| clip1 | clip2 | clip3 |
| --- | --- | --- |
| [![rink clip1](Examples/rink/clip1.jpg)](Examples/rink/clip1.mp4) | [![rink clip2](Examples/rink/clip2.jpg)](Examples/rink/clip2.mp4) | [![rink clip3](Examples/rink/clip3.jpg)](Examples/rink/clip3.mp4) |

## Dots model

**`new_dots.pt`** · `run_models.py` name `dots`

Finds the faceoff dots: the centre dot, the four neutral-zone dots and the four end-zone dots.
Dots are small, so the centre of each box is a precise point on the ice. `homography.py` trusts
them most when it fits the homography, and works out which dot is which from where it sits relative
to the lines and circles the rink model found. About 2 to 3 dots are in view per frame in the
examples.

| clip1 | clip2 | clip3 |
| --- | --- | --- |
| [![dots clip1](Examples/dots/clip1.jpg)](Examples/dots/clip1.mp4) | [![dots clip2](Examples/dots/clip2.jpg)](Examples/dots/clip2.mp4) | [![dots clip3](Examples/dots/clip3.jpg)](Examples/dots/clip3.mp4) |

---

## How the examples were made

The three clips are 10-second (600 frames, 59.94 fps, 1080p) cuts from the full-game broadcast in
`videos/`, which isn't in git. Every model was run with the game's own roster
(`rosters/2026-03-03_MTL_at_SJS.csv`, from `fetch_roster.py --game`):

```bash
python run_models.py <clips folder> --teams SJS MTL \
    --roster rosters/2026-03-03_MTL_at_SJS.csv --save-video --per-model --output <out>
```

That writes `player.mp4`, `puck.mp4`, `rink.mp4`, `dots.mp4` and `detections.json` for each clip.
The number examples are drawn from `detections.json` with player boxes, digit boxes and name tags:

```bash
python CV_Models/Examples/render_numbers.py <clip>.mp4 <out>/<clip>/detections.json number.mp4
```

The videos were then scaled to 1280×720 and re-encoded with
`ffmpeg -vf scale=1280:-2 -c:v libx264 -crf 26 -pix_fmt yuv420p`. Each image is the frame with the
most detections for that model; for the number model it's from the second half of the clip, after
the votes have settled.

## On the Jetson

`python export_engines.py` builds a TensorRT FP16 engine next to each model
(`CV_Models/Models/<name>.engine`), and `run_models.py` uses the engine instead of the `.pt` from
then on. See [jetson/README.md](../jetson/README.md).
