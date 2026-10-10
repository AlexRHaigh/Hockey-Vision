"""Class-aware unidentified players in event analysis; no models needed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import events


def track(tid, cls, role, number=None):
    return {"track_id": tid, "class": cls, "role": role, "team": "A",
            "team_abbrev": "SJS", "jersey_number": number, "player_name": None}


class EventIdentityTests(unittest.TestCase):
    def test_unidentified_class_handover_preserves_both_roles(self):
        tracks = [track(7, "team_a_player", "skater"), track(7, "goalie_a", "goalie")]
        ids, info = events.identify_players("fixture", tracks)
        skater, goalie = ids[(7, "team_a_player")], ids[(7, "goalie_a")]
        self.assertNotEqual(skater, goalie)
        self.assertEqual(info[skater]["role"], "skater")
        self.assertEqual(info[goalie]["role"], "goalie")
        self.assertFalse(info[skater]["identified"])
        self.assertFalse(info[goalie]["identified"])

    def test_goalie_sighting_after_class_handover_still_sets_attack_direction(self):
        tracks = [track(7, "team_a_player", "skater"), track(7, "goalie_a", "goalie")]
        frames = [{"frame": i, "time_s": i / 30, "camera_cut": 0, "view": "side"}
                  for i in range(3)]
        players = [{"frame": 0, "track_id": 7, "class": "team_a_player", "x_ft": 0, "y_ft": 0},
                   {"frame": 1, "track_id": 7, "class": "goalie_a", "x_ft": -87, "y_ft": 0},
                   {"frame": 2, "track_id": 7, "class": "goalie_a", "x_ft": -87, "y_ft": 0}]
        _, _, info, directions = events.analyze_clip("fixture", frames, players, [], tracks)
        self.assertEqual(len(info), 2)
        self.assertEqual(directions, {"A": 1, "B": -1})

    def test_identified_player_remains_shared_across_tracks_and_clips(self):
        tracks = [track(7, "team_a_player", "skater", "72"),
                  track(8, "team_a_player", "skater", "72")]
        ids, info = events.identify_players("first", tracks)
        other_ids, _ = events.identify_players("second", tracks)
        self.assertEqual(ids[(7, "team_a_player")], "SJS #72")
        self.assertEqual(ids[(7, "team_a_player")], ids[(8, "team_a_player")])
        self.assertEqual(ids, other_ids)
        self.assertEqual(len(info), 1)

    def test_unidentified_players_are_still_scoped_to_clip_and_track(self):
        tracks = [track(7, "team_a_player", "skater"), track(8, "team_a_player", "skater")]
        ids, _ = events.identify_players("first", tracks)
        other_ids, _ = events.identify_players("second", tracks)
        self.assertNotEqual(ids[(7, "team_a_player")], ids[(8, "team_a_player")])
        self.assertNotEqual(ids[(7, "team_a_player")], other_ids[(7, "team_a_player")])


if __name__ == "__main__":
    unittest.main()
