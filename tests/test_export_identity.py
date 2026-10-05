"""Regression tests for class-aware export summaries; no model weights needed."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("export_data", Path(__file__).resolve().parents[1] / "export_data.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def row(frame, cls="team_a_player", number="72", team="SJS", name="Player A", x=None):
    return {"frame": frame, "time_s": frame / 30, "track_id": 7, "class": cls,
            "jersey_number": number, "team_abbrev": team, "player_name": name,
            "x_ft": frame * 0.2 if x is None else x, "y_ft": 0.0}


class ExportIdentityTests(unittest.TestCase):
    def test_other_team_handover_keeps_summaries_separate(self):
        a = [row(i) for i in range(6)]
        b = [row(i, "team_b_player", "14", "MTL", "Player B") for i in range(6, 12)]
        result = export.summarise_tracks("fixture", a + b)
        self.assertEqual(len(result), 2)
        self.assertEqual([(r["class"], r["team_abbrev"], r["player_name"], r["jersey_number"])
                          for r in result], [("team_a_player", "SJS", "Player A", "72"),
                                            ("team_b_player", "MTL", "Player B", "14")])
        self.assertEqual([r["frames_detected"] for r in result], [6, 6])

    def test_distance_does_not_bridge_other_team_handover(self):
        a = [row(i, x=0.0) for i in range(6)]
        b = [row(i, "team_b_player", "14", "MTL", "Player B", x=2.0) for i in range(6, 12)]
        result = export.summarise_tracks("fixture", a + b)
        self.assertEqual(len(result), 2)
        self.assertEqual([r["distance_ft"] for r in result], [0.0, 0.0])

    def test_referee_does_not_inherit_player_number_or_name(self):
        result = export.summarise_tracks("fixture", [row(0), row(1, "referee", None, None, None)])
        ref = next(r for r in result if r["class"] == "referee")
        self.assertEqual(ref["role"], "referee")
        self.assertIsNone(ref["jersey_number"])
        self.assertIsNone(ref["player_name"])
        self.assertIsNone(ref["team_abbrev"])

    def test_unchanged_single_class_summary(self):
        result = export.summarise_tracks("fixture", [row(i) for i in range(6)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["frames_detected"], 6)
        self.assertEqual(result[0]["jersey_number"], "72")
        self.assertEqual(result[0]["first_frame"], 0)
        self.assertEqual(result[0]["last_frame"], 5)

    def test_untracked_detections_remain_excluded(self):
        r = row(0)
        r["track_id"] = None
        self.assertEqual(export.summarise_tracks("fixture", [r]), [])


if __name__ == "__main__":
    unittest.main()
