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

`detections.json` lists the digit boxes in `number`. Older readers are kept for comparison, with
their models in `CV_Models/unused_models/jersey_Num Models/` (the old YOLO digit model, `number_model.pt`, is there too):

| `--number-reader` | Model | Notes |
|---|---|---|
| `parseq` | PARSeq, the text recognizer from [Koshkina & Elder's jersey pipeline](https://github.com/mkoshkina/jersey-number-pipeline) (`parseq_jersey.py`, `jersey.ckpt`, their hockey fine-tune), after their legibility classifier (`legibility_resnet34_hockey_20240201.pth`) | with `--teams` it picks the likeliest number the team wears; needs `timm` and `pip install --no-deps git+https://github.com/baudm/parseq.git` |
| `pipeline` | their whole pipeline (`jersey_pipeline.py`): Centroid-ReID filter (`centroid-reid.ckpt`), legibility classifier, ViTPose-H torso crop (`vitpose-h.pth`), PARSeq, their vote | most accurate per reading; ViTPose-H is ~250 GFLOPs per player, ~10x the cost |
| `resnet` | jersey ResNet (`jersey_net.py`, `jersey_model*.pt`), trained per [colab/README.md](colab/README.md) | fast; current weights read NHL jerseys poorly |
| `temporal` | EfficientNet + LSTM from [Hugging Face](https://huggingface.co/Akashpaul123/jersey-number-recognition-temporal) (`temporal_jersey.py`, `jersey_model.pth`) | trained on soccer with ten numbers |
