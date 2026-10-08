"""Play event rules on small synthetic rink scenes; no models or video needed."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("events", ROOT / "events.py")
events = importlib.util.module_from_spec(spec)
spec.loader.exec_module(events)

FPS = 60.0
# (track_id, class, team, role, number)
PLAYERS = {
    "a7": (1, "team_a_player", "A", "skater", "7"),
    "a9": (2, "team_a_player", "A", "skater", "9"),
    "b14": (3, "team_b_player", "B", "skater", "14"),
    "ga": (4, "goalie_a", "A", "goalie", "30"),
    "gb": (5, "goalie_b", "B", "goalie", "31"),
}


class Scene:
    """Frames of player and puck rink positions. Team A's goalie is at the left net, so team A
    attacks right (+x) and team B attacks left."""

    def __init__(self, n):
        self.n = n
        self.pos = {k: [None] * n for k in PLAYERS}
        self.puck = [None] * n
        self.place("ga", range(n), (-87, 0))
        self.place("gb", range(n), (87, 0))

    def place(self, who, frames, xy):
        for f in frames:
            self.pos[who][f] = xy(f) if callable(xy) else xy

    def move(self, who, frames, start, end):
        frames = list(frames)
        for k, f in enumerate(frames):
            u = k / max(1, len(frames) - 1)
            self.pos[who][f] = (start[0] + u * (end[0] - start[0]), start[1] + u * (end[1] - start[1]))

    def carry(self, who, frames, offset=(1.0, 0.0)):
        for f in frames:
            x, y = self.pos[who][f]
            self.puck[f] = (x + offset[0], y + offset[1])

    def analyze(self):
        frame_rows = [{"frame": f, "time_s": f / FPS, "camera_cut": 0, "view": "side"} for f in range(self.n)]
        player_rows = []
        for who, (tid, cls, team, role, num) in PLAYERS.items():
            for f, xy in enumerate(self.pos[who]):
                if xy is not None:
                    player_rows.append({"frame": f, "time_s": f / FPS, "track_id": tid, "class": cls,
                                        "x_ft": xy[0], "y_ft": xy[1]})
        puck_rows = [{"frame": f, "time_s": f / FPS, "x_ft": xy[0], "y_ft": xy[1]}
                     for f, xy in enumerate(self.puck) if xy is not None]
        tracks = [{"track_id": tid, "class": cls, "team": team, "role": role, "jersey_number": num,
                   "team_abbrev": {"A": "SJS", "B": "MTL"}[team], "player_name": None}
                  for tid, cls, team, role, num in PLAYERS.values()]
        return events.analyze_clip("test", frame_rows, player_rows, puck_rows, tracks)


def of_type(evs, type_):
    return [e for e in evs if e["type"] == type_]


class EventTests(unittest.TestCase):
    def test_attacking_direction_from_goalies(self):
        _, _, _, directions = Scene(10).analyze()
        self.assertEqual(directions, {"A": 1, "B": -1})

    def test_shot_saved_by_goalie(self):
        s = Scene(120)
        s.move("a7", range(0, 40), (40, 5), (55, 5))
        s.carry("a7", range(0, 40))
        # Released at (56, 5), the puck flies at the right net at ~90 ft/s ...
        for k, f in enumerate(range(40, 55)):
            s.puck[f] = (56 + 1.5 * (k + 1), 5 - 0.15 * (k + 1))
        # ... and the goalie covers it.
        s.carry("gb", range(60, 90), offset=(-1, 0))
        _, evs, _, _ = s.analyze()
        shots = of_type(evs, "shot")
        self.assertEqual(len(shots), 1)
        self.assertEqual((shots[0]["player_id"], shots[0]["subtype"], shots[0]["target_net"]),
                         ("SJS #7", "saved", "right"))
        self.assertEqual(shots[0]["other_player_id"], "MTL #31")
        self.assertEqual(of_type(evs, "turnover"), [])  # the save is the shot's outcome

    def test_shot_at_own_net_is_not_a_shot(self):
        s = Scene(120)
        s.move("a7", range(0, 40), (-40, 5), (-55, 5))
        s.carry("a7", range(0, 40), offset=(-1, 0))
        for k, f in enumerate(range(40, 55)):
            s.puck[f] = (-56 - 1.5 * (k + 1), 5 - 0.15 * (k + 1))
        _, evs, _, _ = s.analyze()
        self.assertEqual(of_type(evs, "shot"), [])

    def test_good_pass_up_the_ice(self):
        s = Scene(120)
        s.place("a7", range(120), (-10, 0))
        s.place("a9", range(120), (30, 10))
        s.place("b14", range(120), (10, 30))
        s.carry("a7", range(0, 30))
        for k, f in enumerate(range(30, 50)):
            s.puck[f] = (-9 + 2 * (k + 1), 0.5 * (k + 1))
        s.carry("a9", range(50, 90))
        _, evs, _, _ = s.analyze()
        passes = of_type(evs, "pass")
        self.assertEqual(len(passes), 1)
        p = passes[0]
        self.assertEqual((p["player_id"], p["other_player_id"], p["subtype"]), ("SJS #7", "SJS #9", "good"))
        self.assertIn("advanced_puck", p["good_reasons"])
        self.assertIn("beat_defenders", p["good_reasons"])
        self.assertGreater(p["forward_ft"], 30)

    def test_takeaway(self):
        s = Scene(120)
        s.move("a7", range(0, 120), (0, 0), (30, 0))
        s.move("b14", range(0, 120), (6, 2), (33, 2))
        s.carry("a7", range(0, 50))
        s.carry("b14", range(55, 100))
        _, evs, _, _ = s.analyze()
        turnovers = of_type(evs, "turnover")
        self.assertEqual(len(turnovers), 1)
        self.assertEqual((turnovers[0]["subtype"], turnovers[0]["player_id"], turnovers[0]["other_player_id"]),
                         ("takeaway", "MTL #14", "SJS #7"))

    def test_interception(self):
        s = Scene(120)
        s.place("a7", range(120), (-20, 0))
        s.place("b14", range(120), (10, 20))
        s.carry("a7", range(0, 30))
        s.carry("b14", range(60, 90))
        _, evs, _, _ = s.analyze()
        self.assertEqual([e["subtype"] for e in of_type(evs, "turnover")], ["interception"])

    def test_forced_backtrack(self):
        s = Scene(120)
        # SJS #7 skates up the ice, meets MTL #14 and has to turn back.
        s.move("a7", range(0, 60), (0, 0), (30, 0))
        s.move("a7", range(60, 120), (30, 0), (10, 0))
        s.move("b14", range(0, 60), (60, 0), (36, 0))
        s.move("b14", range(60, 120), (36, 0), (18, 0))
        s.carry("a7", range(0, 120))
        _, evs, _, _ = s.analyze()
        plays = of_type(evs, "defensive_play")
        self.assertEqual([(e["subtype"], e["player_id"], e["other_player_id"]) for e in plays],
                         [("forced_backtrack", "MTL #14", "SJS #7")])

    def test_unpressured_carry_has_no_defensive_play(self):
        s = Scene(120)
        s.move("a7", range(0, 120), (0, 0), (30, 0))
        s.place("b14", range(120), (-40, 30))
        s.carry("a7", range(0, 120))
        _, evs, _, _ = s.analyze()
        self.assertEqual(evs, [])

    def test_player_stats_count_events(self):
        s = Scene(120)
        s.move("a7", range(0, 120), (0, 0), (30, 0))
        s.move("b14", range(0, 120), (6, 2), (33, 2))
        s.carry("a7", range(0, 50))
        s.carry("b14", range(55, 100))
        rows, evs, info, _ = s.analyze()
        stats = {r["player_id"]: r for r in events.player_stats(info, evs, rows, {})}
        self.assertEqual(stats["MTL #14"]["takeaways"], 1)
        self.assertEqual(stats["SJS #7"]["turnovers"], 1)
        self.assertEqual(stats["SJS #7"]["possessions"], 1)


if __name__ == "__main__":
    unittest.main()
