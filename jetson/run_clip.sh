#!/usr/bin/env bash
# Run the whole pipeline on one video and write only the data files (no annotated videos):
#   outputs/<clip>/positions.json, positions.csv, homographies.csv (and detections.json)
#
# Usage, from the repo root:  jetson/run_clip.sh videos/<clip>.mp4 [--max-frames N] [--conf X] ...
# Extra arguments go to run_models.py.
set -euo pipefail

video=$1
shift
clip=$(basename "${video%.*}")

python run_models.py "$video" --no-video --no-per-model "$@"
python homography.py "outputs/$clip/detections.json" "$video" --no-video
