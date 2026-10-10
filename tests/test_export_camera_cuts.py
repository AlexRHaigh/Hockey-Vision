"""Distance smoothing must not bridge independently projected camera shots."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_data


class CameraCutDistanceTests(unittest.TestCase):
    def test_same_track_stationary_on_either_side_of_cut_has_no_distance(self):
        rows = [{"frame": i, "time_s": i / 30, "x_ft": 0 if i < 6 else 2,
                 "y_ft": 0, "camera_segment": int(i >= 6)} for i in range(12)]
        distance, seconds = export_data.track_distance(rows)
        self.assertEqual(distance, 0)
        self.assertAlmostEqual(seconds, 10 / 30)

    def test_motion_within_camera_segments_is_preserved(self):
        before = [{"frame": i, "time_s": i / 30, "x_ft": i * 0.2, "y_ft": 0,
                   "camera_segment": 0} for i in range(6)]
        after = [{"frame": i, "time_s": i / 30, "x_ft": 10 + i * 0.2, "y_ft": 0,
                  "camera_segment": 1} for i in range(6, 12)]
        expected_distance = export_data.track_distance(before)[0] + export_data.track_distance(after)[0]
        distance, _ = export_data.track_distance(before + after)
        self.assertGreater(distance, 0)
        self.assertAlmostEqual(distance, expected_distance)

    def test_export_retains_cut_even_when_player_is_missing_on_cut_frame(self):
        with tempfile.TemporaryDirectory() as root:
            clip = Path(root) / "fixture"
            clip.mkdir()
            frames, positions = [], []
            for i in range(12):
                player = {"class": "team_a_player", "confidence": 0.9,
                          "box_xyxy": [0, 0, 20, 50], "track_id": 7, "jersey_number": "72"}
                detected = [] if i == 6 else [player]
                frames.append({"frame": i, "time_s": i / 30, "detections": {"player": detected}})
                projected = [{"box_xyxy": player["box_xyxy"], "rink_xy": [0 if i < 6 else 2, 0]}]
                positions.append({"frame": i, "cut": i == 6, "view": "side",
                                  "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  "keypoints_used": [], "players": projected if detected else [], "puck": None})
            (clip / "detections.json").write_text(json.dumps(frames))
            (clip / "positions.json").write_text(json.dumps(positions))
            tables, *_ = export_data.export_clip(clip)
            self.assertEqual(tables["frames.csv"][6]["camera_cut"], 1)
            self.assertEqual(tables["tracks.csv"][0]["frames_detected"], 11)
            self.assertEqual(tables["tracks.csv"][0]["distance_ft"], 0)
            self.assertEqual(tables["tracks.csv"][0]["mean_speed_ft_s"], 0)


if __name__ == "__main__":
    unittest.main()
