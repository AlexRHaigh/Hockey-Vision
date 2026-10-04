# Hockey-Vision

To run on an NVIDIA Jetson Orin Nano, see [jetson/README.md](jetson/README.md).

## Pipeline

1. `run_models.py <video>` runs the detection models and writes `outputs/<clip>/detections.json`.
2. `homography.py outputs/<clip>/detections.json <video>` maps players and the puck onto the rink
   (`positions.json`, `positions.csv`, `homographies.csv`, and `side_by_side.mp4`).
3. `export_data.py outputs/<clip>` writes analysis-ready tables to `outputs/<clip>/export/`:
   `players.csv`, `puck.csv`, `frames.csv`, `tracks.csv` and `metadata.json`
   (see [jetson/README.md](jetson/README.md#exported-tables)).

## Jersey numbers

`run_models.py` reads jersey numbers with PARSeq, the text recognizer from
[Koshkina & Elder's jersey pipeline](https://github.com/mkoshkina/jersey-number-pipeline)
(`parseq_jersey.py`, `CV_Models/jersey_Num Models/jersey.ckpt`: their hockey fine-tune). Their legibility classifier
(`CV_Models/jersey_Num Models/legibility_resnet34_hockey_20240201.pth`) first checks each whole player crop for a
readable number, and only those are read; without it PARSeq reads every player and answers "4" for
many unreadable ones. PARSeq reads a fixed crop of the player's box (22-78% across, 15-45% down: where their pose model finds the
shoulders-to-hips torso on our footage), restricted to digits, and each player track's readings are
combined with their vote (readings under 0.2 confidence ignored, two-digit numbers favoured). It
needs `timm` and the PARSeq code, which isn't on PyPI:

```bash
pip install -r requirements.txt
pip install --no-deps git+https://github.com/baudm/parseq.git
```

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

PARSeq then reads the number its digit probabilities favour among that team's skaters' (or
goalies') numbers, so impossible readings and dropped digits ("7" for 72) give way to numbers
someone actually wears. `detections.json`, `positions.csv` and the exported `players.csv` /
`tracks.csv` gain `team_abbrev` / `player_name`. On our two test clips this removed every wrong
number (9 shown, 9 right) while reading as many players. The roster CSV has one row per player:
`team, team_name, number, player, position, player_id, as_of`; refetch it as rosters change.

Without `CV_Models/jersey_Num Models/jersey.ckpt`, or with `--number-reader yolo`, numbers are read by
`number_model` instead, which detects single digits and joins them (only then does
`detections.json` have digit boxes in `number`). Other readers, kept for comparison:

| `--number-reader` | Model | Notes |
|---|---|---|
| `pipeline` | their whole pipeline (`jersey_pipeline.py`): Centroid-ReID filter (`centroid-reid.ckpt`), legibility classifier, ViTPose-H torso crop (`vitpose-h.pth`), PARSeq, their vote | most accurate per reading; ViTPose-H is ~250 GFLOPs per player, ~10x the cost |
| `resnet` | jersey ResNet (`jersey_net.py`, `CV_Models/jersey_Num Models/jersey_model*.pt`), trained per [colab/README.md](colab/README.md) | fast; current weights read NHL jerseys poorly |
| `temporal` | EfficientNet + LSTM from [Hugging Face](https://huggingface.co/Akashpaul123/jersey-number-recognition-temporal) (`temporal_jersey.py`, `CV_Models/jersey_Num Models/jersey_model.pth`) | trained on soccer with ten numbers |
