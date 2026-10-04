#!/usr/bin/env bash
# Run the whole pipeline on one video and write only the data files (no annotated videos):
#   outputs/<clip>/positions.json, positions.csv, homographies.csv (and detections.json)
#   outputs/<clip>/export/players.csv, puck.csv, frames.csv, tracks.csv, metadata.json
#
# Usage, from the repo root:  jetson/run_clip.sh videos/<clip>.mp4 [--max-frames N] [--conf X] ...
# Extra arguments go to run_models.py, e.g. --teams SJS MTL (team_a, team_b) to read numbers
# from the teams' rosters and name the players (see fetch_roster.py). Jersey numbers are read with the
# YOLO number model (CV_Models/Models/new_nums.pt); a player whose number is settled is re-read every 5th frame
# (--jersey-stride, 1 to read every frame).
set -euo pipefail

video=$1
shift
clip=$(basename "${video%.*}")

python run_models.py "$video" --no-video --no-per-model "$@"
python homography.py "outputs/$clip/detections.json" "$video" --no-video
python export_data.py "outputs/$clip"
