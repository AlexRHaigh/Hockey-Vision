#!/usr/bin/env bash
# Run the whole pipeline on one video and write only the data files (no annotated videos):
#   outputs/<clip>/positions.json, positions.csv, homographies.csv (and detections.json)
#   outputs/<clip>/export/players.csv, puck.csv, frames.csv, tracks.csv, metadata.json
#
# Usage, from the repo root:  jetson/run_clip.sh videos/<clip>.mp4 [--max-frames N] [--conf X] ...
# Extra arguments go to run_models.py (and override the defaults below).
# Settled jersey numbers are re-read every 5th frame rather than every frame, the biggest saving
# on the Jetson; pass --jersey-stride 1 to read them every frame.
set -euo pipefail

video=$1
shift
clip=$(basename "${video%.*}")

python run_models.py "$video" --no-video --no-per-model --jersey-stride 5 "$@"
python homography.py "outputs/$clip/detections.json" "$video" --no-video
python export_data.py "outputs/$clip"
