"""Turn run_models.py / homography.py outputs into analysis-ready tables.

    python export_data.py outputs/<clip>               # one clip  -> outputs/<clip>/export/
    python export_data.py outputs/<a> outputs/<b>      # several   -> outputs/export/
    python export_data.py outputs                      # every clip folder in outputs/
    python export_data.py outputs --output exports/game1

Each clip folder needs detections.json (from run_models.py); positions.json (from homography.py)
adds the rink coordinates. Written to the output folder, every table with a `clip` column so
several clips can be exported together:
    players.csv    one row per player detection per frame: track id, team, role, jersey number,
                   image box, and rink x, y in feet (empty when the frame had no usable fit)
    puck.csv       one row per frame with a puck detection: image box and rink x, y
    frames.csv     one row per frame: whether the rink fit worked, counts of what was seen
    tracks.csv     one row per player track: team, jersey number, when it was seen, distance skated
    metadata.json  source files, frame rate, coordinate system and a description of every column

Only the standard library is used, so this runs anywhere the output folders are copied to.

Rink coordinates are feet from the centre dot: x along the rink (-100 left end boards to 100
right), y across it (-42.5 near boards to 42.5 far boards); see rink.py. A player's position is
where their skates (the bottom-centre of their box) meet the ice.

Distance and speed are computed from the rink positions smoothed over a few frames, within runs
of frames where the track was on the rink. Steps faster than a skater can move are projection
errors and are left out.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Player model classes -> (team, role).
PLAYER_ROLES = {
    "team_a_player": ("A", "skater"),
    "goalie_a": ("A", "goalie"),
    "team_b_player": ("B", "skater"),
    "goalie_b": ("B", "goalie"),
    "referee": ("", "referee"),
}

SMOOTH_WINDOW = 5      # frames in the centred moving average applied to a track's rink positions
MAX_GAP_FRAMES = 5     # a track missing for more frames than this starts a new run (no distance across it)
MAX_SPEED_FT_S = 40.0  # NHL skaters top out around 35 ft/s; faster steps are bad projections

COLUMNS = {
    "players.csv": {
        "clip": "clip name (the output folder name)",
        "frame": "frame index in the video, from 0",
        "time_s": "seconds from the start of the video",
        "track_id": "player tracker id, stable across frames within a clip (empty if untracked)",
        "class": "player model class",
        "team": "A or B (empty for referees)",
        "role": "skater, goalie or referee",
        "jersey_number": "jersey number shown for the track at this frame (empty if not read yet)",
        "team_abbrev": "the player's team, e.g. SJS (only with run_models.py --teams)",
        "player_name": "the player wearing jersey_number for that team (only with run_models.py --teams)",
        "confidence": "player detection confidence",
        "box_x1": "image box left, px", "box_y1": "image box top, px",
        "box_x2": "image box right, px", "box_y2": "image box bottom, px",
        "x_ft": "rink x of the skates, ft (empty if no usable rink fit this frame)",
        "y_ft": "rink y of the skates, ft (empty if no usable rink fit this frame)",
    },
    "puck.csv": {
        "clip": "clip name",
        "frame": "frame index",
        "time_s": "seconds from the start of the video",
        "confidence": "confidence of the most confident puck detection",
        "box_x1": "image box left, px", "box_y1": "image box top, px",
        "box_x2": "image box right, px", "box_y2": "image box bottom, px",
        "x_ft": "rink x, ft (empty if no usable rink fit or projected off the ice)",
        "y_ft": "rink y, ft (empty if no usable rink fit or projected off the ice)",
    },
    "frames.csv": {
        "clip": "clip name",
        "frame": "frame index",
        "time_s": "seconds from the start of the video",
        "rink_fit": "1 if this frame has a usable image -> rink homography, else 0",
        "rejected": "why a fit was thrown out (players_clustered, player_scale), empty otherwise",
        "keypoints_used": "number of rink landmarks the fit used",
        "players_detected": "player boxes in the frame",
        "players_on_rink": "players with a rink position",
        "puck_detected": "1 if a puck was detected",
        "puck_on_rink": "1 if the puck has a rink position",
    },
    "tracks.csv": {
        "clip": "clip name",
        "track_id": "player tracker id",
        "class": "most frequent player model class over the track",
        "team": "A or B (empty for referees)",
        "role": "skater, goalie or referee",
        "jersey_number": "most frequently shown jersey number (empty if never read)",
        "team_abbrev": "the track's team, e.g. SJS (only with run_models.py --teams)",
        "player_name": "the player wearing jersey_number for that team (only with run_models.py --teams)",
        "first_frame": "first frame the track was detected",
        "last_frame": "last frame the track was detected",
        "first_time_s": "time of first_frame, s",
        "last_time_s": "time of last_frame, s",
        "frames_detected": "frames the track was detected in",
        "frames_on_rink": "frames the track had a rink position",
        "mean_x_ft": "mean rink x, ft",
        "mean_y_ft": "mean rink y, ft",
        "distance_ft": "distance covered on the rink, ft (smoothed, see metadata notes)",
        "mean_speed_ft_s": "distance_ft over the time it was measured across, ft/s",
    },
}


def load_frames(path):
    frames = json.loads(path.read_text())
    if isinstance(frames, dict):  # single image output of run_models.py
        frames = [{"frame": 0, "time_s": 0.0, "detections": frames}]
    return frames


def find_clips(paths):
    """Clip folders (containing detections.json) among `paths`, or directly inside them."""
    clips = []
    for p in paths:
        if (p / "detections.json").exists():
            clips.append(p)
        elif p.is_dir():
            found = sorted(d for d in p.iterdir() if (d / "detections.json").exists())
            if not found:
                print(f"No detections.json in {p} or its subfolders, skipping")
            clips.extend(found)
        else:
            print(f"Not a folder: {p}, skipping")
    return clips


def _box_key(box):
    return tuple(round(v, 1) for v in box)


def _blank(v):
    return "" if v is None else v


def export_clip(clip_dir):
    """Returns (tables, info) for one clip: {table_name: [row dict, ...]} and metadata."""
    clip = clip_dir.name
    frames = load_frames(clip_dir / "detections.json")
    positions_path = clip_dir / "positions.json"
    positions = {}
    if positions_path.exists():
        positions = {p["frame"]: p for p in json.loads(positions_path.read_text())}
        if any(pl.get("box_xyxy") is None for p in positions.values() for pl in p["players"]):
            sys.exit(f"{positions_path} is from an older homography.py without player boxes; "
                     f"re-run homography.py (with --no-video it's quick) and export again")
    else:
        print(f"  {clip}: no positions.json, exporting without rink coordinates "
              f"(run homography.py first to get them)")

    players, pucks, frame_rows = [], [], []
    for f in frames:
        idx, t, dets = f["frame"], f["time_s"], f["detections"]
        pos = positions.get(idx)
        on_rink = {_box_key(pl["box_xyxy"]): pl for pl in pos["players"]} if pos else {}

        n_players = n_on_rink = 0
        for d in dets.get("player", []):
            if d["class"] not in PLAYER_ROLES:
                continue
            team, role = PLAYER_ROLES[d["class"]]
            rink_xy = on_rink.get(_box_key(d["box_xyxy"]), {}).get("rink_xy") or [None, None]
            n_players += 1
            n_on_rink += rink_xy[0] is not None
            x1, y1, x2, y2 = d["box_xyxy"]
            players.append({
                "clip": clip, "frame": idx, "time_s": t, "track_id": d.get("track_id"),
                "class": d["class"], "team": team, "role": role,
                "jersey_number": d.get("jersey_number"), "team_abbrev": d.get("team"),
                "player_name": d.get("player_name"), "confidence": d["confidence"],
                "box_x1": x1, "box_y1": y1, "box_x2": x2, "box_y2": y2,
                "x_ft": rink_xy[0], "y_ft": rink_xy[1],
            })

        # Same choice as homography.project_puck: the most confident puck is the puck.
        puck_dets = [d for d in dets.get("puck", []) if d["class"] == "puck"]
        puck_xy = (pos or {}).get("puck") or {}
        if puck_dets:
            best = max(puck_dets, key=lambda d: d["confidence"])
            x1, y1, x2, y2 = best["box_xyxy"]
            xy = puck_xy.get("rink_xy") or [None, None]
            pucks.append({"clip": clip, "frame": idx, "time_s": t, "confidence": best["confidence"],
                          "box_x1": x1, "box_y1": y1, "box_x2": x2, "box_y2": y2,
                          "x_ft": xy[0], "y_ft": xy[1]})

        frame_rows.append({
            "clip": clip, "frame": idx, "time_s": t,
            "rink_fit": int(bool(pos and pos["homography"] is not None)),
            "rejected": (pos or {}).get("rejected"),
            "keypoints_used": len((pos or {}).get("keypoints_used", [])),
            "players_detected": n_players, "players_on_rink": n_on_rink,
            "puck_detected": int(bool(puck_dets)), "puck_on_rink": int(bool(puck_xy)),
        })

    info = {
        "clip": clip,
        "source": {"detections": str(clip_dir / "detections.json"),
                   "positions": str(positions_path) if positions else None},
        "frames": len(frames),
        "fps": _fps(frames),
        "frames_with_rink_fit": sum(r["rink_fit"] for r in frame_rows),
    }
    return {"players.csv": players, "puck.csv": pucks, "frames.csv": frame_rows,
            "tracks.csv": summarise_tracks(clip, players)}, info


def _fps(frames):
    if len(frames) < 2 or frames[1]["time_s"] <= frames[0]["time_s"]:
        return None
    return round((frames[1]["frame"] - frames[0]["frame"]) / (frames[1]["time_s"] - frames[0]["time_s"]), 3)


def _smooth(points):
    """Centred moving average of [(x, y)], the window shrinking at the ends."""
    half = SMOOTH_WINDOW // 2
    out = []
    for i in range(len(points)):
        win = points[max(0, i - half):i + half + 1]
        out.append((sum(p[0] for p in win) / len(win), sum(p[1] for p in win) / len(win)))
    return out


def track_distance(rows):
    """(distance_ft, seconds it was measured over) from one track's rows, sorted by frame."""
    on_rink = [r for r in rows if r["x_ft"] is not None]
    runs, run = [], []
    for r in on_rink:
        if run and r["frame"] - run[-1]["frame"] > MAX_GAP_FRAMES:
            runs.append(run)
            run = []
        run.append(r)
    if run:
        runs.append(run)

    distance = seconds = 0.0
    for run in runs:
        xy = _smooth([(r["x_ft"], r["y_ft"]) for r in run])
        for (a, pa), (b, pb) in zip(zip(run, xy), zip(run[1:], xy[1:])):
            dt = b["time_s"] - a["time_s"]
            step = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
            if dt > 0 and step / dt <= MAX_SPEED_FT_S:
                distance += step
                seconds += dt
    return distance, seconds


def summarise_tracks(clip, players):
    by_track = defaultdict(list)
    for r in players:
        if r["track_id"] is not None:
            by_track[r["track_id"]].append(r)
    out = []
    for tid in sorted(by_track):
        rows = sorted(by_track[tid], key=lambda r: r["frame"])
        cls = Counter(r["class"] for r in rows).most_common(1)[0][0]
        team, role = PLAYER_ROLES[cls]
        numbers = Counter(r["jersey_number"] for r in rows if r["jersey_number"])
        number = numbers.most_common(1)[0][0] if numbers else None
        named = [r for r in rows if r["jersey_number"] == number and r["player_name"]]
        abbrevs = Counter(r["team_abbrev"] for r in rows if r["team_abbrev"])
        on_rink = [r for r in rows if r["x_ft"] is not None]
        distance, seconds = track_distance(rows)
        out.append({
            "clip": clip, "track_id": tid, "class": cls, "team": team, "role": role,
            "jersey_number": number,
            "team_abbrev": abbrevs.most_common(1)[0][0] if abbrevs else None,
            "player_name": named[0]["player_name"] if named else None,
            "first_frame": rows[0]["frame"], "last_frame": rows[-1]["frame"],
            "first_time_s": rows[0]["time_s"], "last_time_s": rows[-1]["time_s"],
            "frames_detected": len(rows), "frames_on_rink": len(on_rink),
            "mean_x_ft": round(sum(r["x_ft"] for r in on_rink) / len(on_rink), 2) if on_rink else None,
            "mean_y_ft": round(sum(r["y_ft"] for r in on_rink) / len(on_rink), 2) if on_rink else None,
            "distance_ft": round(distance, 1) if on_rink else None,
            "mean_speed_ft_s": round(distance / seconds, 2) if seconds > 0 else None,
        })
    return out


def write_table(path, columns, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([_blank(r[c]) for c in columns])


def main():
    parser = argparse.ArgumentParser(description="Export Hockey-Vision outputs as analysis-ready CSV tables.")
    parser.add_argument("clips", type=Path, nargs="+",
                        help="Clip output folders (outputs/<clip>), or a folder of them")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output folder (default: <clip>/export for one clip, outputs/export for several)")
    args = parser.parse_args()

    clips = find_clips(args.clips)
    if not clips:
        sys.exit("No clip folders to export")
    out_dir = args.output or (clips[0] / "export" if len(clips) == 1 else Path("outputs") / "export")
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {name: [] for name in COLUMNS}
    infos = []
    for clip_dir in clips:
        clip_tables, info = export_clip(clip_dir)
        for name, rows in clip_tables.items():
            tables[name].extend(rows)
        infos.append(info)
        print(f"  {info['clip']}: {info['frames']} frames, {info['frames_with_rink_fit']} with a rink fit, "
              f"{len(clip_tables['tracks.csv'])} player tracks")

    for name, columns in COLUMNS.items():
        write_table(out_dir / name, list(columns), tables[name])
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clips": infos,
        "coordinates": {
            "units": "feet",
            "origin": "centre ice dot",
            "x": "along the rink, -100 (left end boards) to 100 (right end boards)",
            "y": "across the rink, -42.5 (near boards, camera side) to 42.5 (far boards)",
            "player_point": "bottom-centre of the player's box (skates on the ice)",
        },
        "notes": {
            "distance": f"positions smoothed with a {SMOOTH_WINDOW}-frame centred moving average; no distance "
                        f"across gaps over {MAX_GAP_FRAMES} frames; steps over {MAX_SPEED_FT_S} ft/s dropped",
            "track_id": "unique within a clip only; the tracker can hand an id to another player in a scrum",
        },
        "tables": COLUMNS,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {', '.join(COLUMNS)} and metadata.json to {out_dir}/")


if __name__ == "__main__":
    main()
