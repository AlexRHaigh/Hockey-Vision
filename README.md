# Hockey-Vision

To run on an NVIDIA Jetson Orin Nano, see [jetson/README.md](jetson/README.md).

## Pipeline

1. `run_models.py <video>` runs the detection models and writes `outputs/<clip>/detections.json`.
2. `homography.py outputs/<clip>/detections.json <video>` maps players and the puck onto the rink
   (`positions.json`, `positions.csv`, `homographies.csv`, and `side_by_side.mp4`).
3. `export_data.py outputs/<clip>` writes analysis-ready tables to `outputs/<clip>/export/`:
   `players.csv`, `puck.csv`, `frames.csv`, `tracks.csv` and `metadata.json`
   (see [jetson/README.md](jetson/README.md#exported-tables)).
