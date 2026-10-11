"""Detection-only export keeps player identity and presence without rink fits."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import events
import export_data


def make_clip(root, positions=None):
    clip = Path(root) / "fixture"
    clip.mkdir()
    player = {"class": "team_a_player", "confidence": 0.9, "box_xyxy": [0, 0, 20, 50],
              "track_id": 7, "jersey_number": "72", "team": "SJS", "player_name": "Player A"}
    frames = [{"frame": i, "time_s": i / 30, "detections": {"player": [player]}}
              for i in range(12)]
    (clip / "detections.json").write_text(json.dumps(frames))
    if positions is not None:
        (clip / "positions.json").write_text(json.dumps(positions))
    return clip


class DetectionOnlyExportTests(unittest.TestCase):
    def test_absent_and_empty_positions_keep_identity_and_detected_time(self):
        for positions in (None, []):
            with self.subTest(positions=positions), tempfile.TemporaryDirectory() as root:
                tables, evs, info, presence, metadata = export_data.export_clip(make_clip(root, positions))
                stats = events.player_stats(info, evs, tables["possessions.csv"], presence)
                self.assertEqual(len(stats), 1)
                player = stats[0]
                self.assertEqual(player["player_id"], "SJS #72")
                self.assertEqual(player["player_name"], "Player A")
                self.assertEqual(player["frames_detected"], 12)
                self.assertEqual(player["time_detected_s"], 0.4)
                self.assertEqual(player["time_on_rink_s"], 0)
                self.assertEqual(player["distance_ft"], 0)
                self.assertEqual(player["shots"], 0)
                self.assertEqual(evs, [])
                self.assertEqual(tables["possessions.csv"], [])
                report = export_data.game_report([metadata], stats, evs)
                self.assertEqual(report["players"][0]["player_id"], "SJS #72")

    def test_cli_writes_detection_only_player_stats_and_report(self):
        with tempfile.TemporaryDirectory() as root:
            clip = make_clip(root)
            subprocess.run([sys.executable, str(ROOT / "export_data.py"), str(clip)],
                           check=True, capture_output=True, text=True)
            with (clip / "export" / "player_stats.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["player_id"], "SJS #72")
            self.assertEqual(rows[0]["frames_detected"], "12")
            report = json.loads((clip / "export" / "game_report.json").read_text())
            self.assertEqual(report["players"][0]["player_id"], "SJS #72")
            self.assertEqual(report["teams"]["SJS"]["players_identified"], 1)


if __name__ == "__main__":
    unittest.main()
