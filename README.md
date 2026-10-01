# Hockey-Vision

To run on an NVIDIA Jetson Orin Nano, see [jetson/README.md](jetson/README.md).

## Pipeline

1. `run_models.py <video>` runs the detection models and writes `outputs/<clip>/detections.json`.
2. `homography.py outputs/<clip>/detections.json <video>` maps players and the puck onto the rink
   (`positions.json`, `positions.csv`, `homographies.csv`, and `side_by_side.mp4`).
3. `export_data.py outputs/<clip>` writes analysis-ready tables to `outputs/<clip>/export/`:
   `players.csv`, `puck.csv`, `frames.csv`, `tracks.csv` and `metadata.json`
   (see [jetson/README.md](jetson/README.md#exported-tables)).

## Jersey number ResNet (optional)

`run_models.py --number-reader resnet` reads each player's whole number from a crop of their torso
with a ResNet (`jersey_net.py`), instead of detecting single digits with `number_model` and joining
them. It feeds the same per-track number votes, so `--jersey-stride` and everything downstream
work as before (`detections.json` has no digit boxes in `number` with it).

To train it, starting from a YOLO digit dataset (one box per digit, classes `0`-`9`):

1. `python make_jersey_labels.py path/to/dataset` turns the digit boxes into whole-number labels
   (`labels.csv`). Pass `--torso` if the images are already torso crops rather than whole players.
   Keep each game in a single split (train / valid / test), or validation accuracy means nothing.
2. `python train_jersey.py path/to/dataset` trains a ResNet-34 (`--arch resnet18` for about half
   the cost on the Jetson; `--torso` as above) on a Mac or an NVIDIA PC, and saves the best epoch to
   `CV_Models/jersey_model.pt`. Watch the `numbered` accuracy: crops that show a number.
3. Copy `CV_Models/jersey_model.pt` to the Jetson and build its engine with
   `python export_engines.py --models jersey` (see [jetson/README.md](jetson/README.md)).

Compare the two readers on the same clips by the number shown per track (`tracks.csv`), not by
per-crop accuracy.
