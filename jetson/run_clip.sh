#!/usr/bin/env bash
# Run the whole pipeline on one video and write the data files (no annotated videos):
#   outputs/<clip>/detections.json, positions.json, positions.csv, homographies.csv
#   outputs/<clip>/export/  tables, play events and game_report.json (see export_data.py)
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

python run_models.py "$video" "$@"
python homography.py "outputs/$clip/detections.json" "$video"
python export_data.py "outputs/$clip"
