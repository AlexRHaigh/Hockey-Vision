#!/usr/bin/env bash
# Run the whole pipeline on one video and write only the data files (no annotated videos):
#   outputs/<clip>/positions.json, positions.csv, homographies.csv (and detections.json)
#   outputs/<clip>/export/players.csv, puck.csv, frames.csv, tracks.csv, metadata.json
#
# Usage, from the repo root:  jetson/run_clip.sh videos/<clip>.mp4 [--max-frames N] [--conf X] ...
# Extra arguments go to run_models.py, e.g. --teams SJS MTL (team_a, team_b) to read numbers
# from the teams' rosters and name the players (see fetch_roster.py). Jersey numbers are read with PARSeq (CV_Models/jersey_Num Models/jersey.ckpt)
# on every frame; with the costlier --number-reader yolo or pipeline, --jersey-stride 3 saves time.
set -euo pipefail

video=$1
shift
clip=$(basename "${video%.*}")

python run_models.py "$video" --no-video --no-per-model "$@"
python homography.py "outputs/$clip/detections.json" "$video" --no-video
python export_data.py "outputs/$clip"
