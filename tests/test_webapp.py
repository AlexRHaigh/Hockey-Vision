"""The web app: game scores, form and trade calls, pool assessment, importing exports, and the HTTP
API end to end on a temporary database. No models or video needed."""
import csv
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from webapp import api, db, ingest, scoring, server  # noqa: E402

CAL = scoring.DEFAULT_CALIBRATION
ROSTER = [
    {"team": "SJS", "team_name": "San Jose Sharks", "number": 71, "player": "Macklin Celebrini", "position": "Center", "player_id": 1, "as_of": "2026-10-01"},
    {"team": "SJS", "team_name": "San Jose Sharks", "number": 48, "player": "Will Smith", "position": "Center", "player_id": 2, "as_of": "2026-10-01"},
    {"team": "SJS", "team_name": "San Jose Sharks", "number": 30, "player": "Yaroslav Askarov", "position": "Goalie", "player_id": 3, "as_of": "2026-10-01"},
    {"team": "MTL", "team_name": "Montréal Canadiens", "number": 14, "player": "Nick Suzuki", "position": "Center", "player_id": 4, "as_of": "2026-10-01"},
    {"team": "MTL", "team_name": "Montréal Canadiens", "number": 8, "player": "Mike Matheson", "position": "Defense", "player_id": 5, "as_of": "2026-10-01"},
    {"team": "MTL", "team_name": "Montréal Canadiens", "number": 35, "player": "Sam Montembeault", "position": "Goalie", "player_id": 6, "as_of": "2026-10-01"},
]


def stats_row(team, number, name, role="skater", seconds=300, **counts):
    return {"player_id": f"{team} #{number}", "team": "A", "team_abbrev": team, "jersey_number": str(number),
            "player_name": name, "role": role, "identified": True, "time_detected_s": seconds, **counts}


def write_roster(path, rows=ROSTER):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["team", "team_name", "number", "player", "position", "player_id", "as_of"])
        w.writeheader()
        w.writerows(rows)


class ScoringTests(unittest.TestCase):
    def test_typical_game_scores_50(self):
        # Takeaways worth exactly the typical points per minute over 10 minutes.
        stats = {"time_detected_s": 600, "takeaways": CAL["skater"][0] * 10 / scoring.SKATER_WEIGHTS["takeaways"][1]}
        self.assertAlmostEqual(scoring.game_score(stats, "skater", CAL)["score"], 50, delta=0.2)

    def test_better_play_scores_higher_and_turnovers_lower(self):
        base = {"time_detected_s": 300, "passes_completed": 4}
        good = scoring.game_score({**base, "takeaways": 3, "shots_on_net": 2, "shots": 2}, "skater", CAL)["score"]
        bad = scoring.game_score({**base, "turnovers": 4}, "skater", CAL)["score"]
        self.assertGreater(good, 60)
        self.assertLess(bad, 40)

    def test_short_appearances_are_shrunk_towards_typical(self):
        brief = scoring.game_score({"time_detected_s": 10, "shots": 1, "shots_on_net": 1, "possible_goals": 1}, "skater", CAL)
        long = scoring.game_score({"time_detected_s": 600, "shots": 30, "shots_on_net": 30, "possible_goals": 30}, "skater", CAL)
        self.assertEqual(brief["confidence"], "low")
        self.assertLess(brief["score"], 90)
        self.assertGreater(long["score"], brief["score"])

    def test_scores_stay_in_range(self):
        for stats in ({"time_detected_s": 60, "turnovers": 50}, {"time_detected_s": 60, "possible_goals": 50}, {}):
            score = scoring.game_score(stats, "skater", CAL)["score"]
            self.assertTrue(0 <= score <= 100, score)

    def test_goalie_uses_goalie_weights(self):
        s = scoring.game_score({"time_detected_s": 900, "saves": 20, "goals_against": 1}, "goalie", CAL)
        self.assertIn("saves", [c["stat"] for c in s["components"]])
        self.assertNotIn("shots", [c["stat"] for c in s["components"]])

    def test_calibration_needs_enough_rows(self):
        self.assertEqual(scoring.calibrate([("skater", 10, 5)] * 5), CAL)
        cal = scoring.calibrate([("skater", 3 * m, m) for m in range(1, 40)])
        self.assertAlmostEqual(cal["skater"][0], 3.0)
        self.assertEqual(cal["skater"][1], scoring.MIN_SPREAD["skater"])

    def test_form_weights_recent_games(self):
        games = [{"score": 80, "minutes": 5}, {"score": 40, "minutes": 5}, {"score": 40, "minutes": 5}, {"score": 40, "minutes": 5}]
        f = scoring.form(games)
        self.assertGreater(f["form"], f["season"])
        self.assertGreater(f["trend"], 0)
        self.assertEqual(f["last"], 80)

    def test_recommendations(self):
        def rec(scores):
            return scoring.recommendation(scoring.form([{"score": v, "minutes": 5} for v in scores]))["action"]
        self.assertEqual(rec([]), "hold")
        self.assertEqual(rec([20, 20]), "hold")
        self.assertEqual(rec([30, 30, 30]), "trade")
        self.assertEqual(rec([44, 44, 44]), "watch")
        self.assertEqual(rec([55, 55, 55]), "keep")
        self.assertEqual(rec([50, 50, 50, 75, 75, 75]), "watch")  # sliding

    def player(self, pid, position, scores):
        f = scoring.form([{"score": v, "minutes": 5} for v in scores])
        return {"player_id": pid, "name": f"P{pid}", "team": "SJS", "position": position, "form": f,
                "recommendation": scoring.recommendation(f)}

    def test_strong_pool_holds(self):
        pool = [self.player(i, "Center", [65] * 5) for i in range(3)]
        others = [self.player(10 + i, "Center", [50] * 5) for i in range(10)]
        a = scoring.assess_pool(pool, others, {"F": 3, "D": 0, "G": 0})
        self.assertEqual(a["verdict"], "hold")
        self.assertEqual(a["grade"], "A")
        self.assertEqual(a["upgrades"], [])

    def test_weak_pool_should_trade_with_distinct_upgrades(self):
        pool = [self.player(i, "Center", [30] * 5) for i in range(3)] + [self.player(3, "Goalie", [60] * 5)]
        others = [self.player(10 + i, "Center", [70 + i] * 5) for i in range(9)] + [self.player(30, "Defense", [90] * 5)]
        a = scoring.assess_pool(pool, others, {"F": 3, "D": 1, "G": 1})
        self.assertEqual(a["verdict"], "active")
        self.assertEqual(a["counts"]["trade"], 3)
        self.assertEqual(len(a["upgrades"]), 3)
        targets = [t["player_id"] for u in a["upgrades"] for t in u["targets"]]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertNotIn(30, targets)  # a defenseman is no upgrade for a centre
        self.assertEqual([d["gap"] for d in a["depth"]], [0, 1, 0])

    def test_empty_pool(self):
        self.assertEqual(scoring.assess_pool([], [], {"F": 1})["verdict"], "empty")


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.con = db.connect(self.dir / "t.db")
        write_roster(self.dir / "roster.csv")
        db.load_roster(self.con, self.dir / "roster.csv")

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def write_export(self, rows, name="export"):
        out = self.dir / name
        out.mkdir()
        columns = sorted({k for r in rows for k in r})
        with open(out / "player_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows([{k: ("true" if v is True else "false" if v is False else v) for k, v in r.items()} for r in rows])
        return out

    def test_import_matches_players_and_credits_goals_against(self):
        rows = [stats_row("SJS", 71, "Macklin Celebrini", shots=3, possible_goals=1),
                stats_row("SJS", 30, "Yaroslav Askarov", "goalie", 1200, saves=12),
                stats_row("MTL", 35, "Sam Montembeault", "goalie", 1100, saves=9),
                stats_row("MTL", 99, "Not On Roster"),
                {**stats_row("MTL", 14, "Nick Suzuki"), "identified": False}]
        export = self.write_export(rows)
        game_id, matched, skipped = ingest.import_export(self.con, export, date="2026-10-10", away="MTL", home="SJS")
        self.assertEqual((matched, skipped), (3, 2))
        stored = {r["player_id"]: json.loads(r["stats"]) for r in self.con.execute("SELECT * FROM game_stats")}
        self.assertEqual(stored[1]["shots"], 3)
        self.assertEqual(stored[6]["goals_against"], 1)  # MTL's goalie, SJS scored
        self.assertEqual(stored[3]["goals_against"], 0)
        game = self.con.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        self.assertEqual((game["label"], game["date"]), ("MTL at SJS", "2026-10-10"))

    def test_reimport_replaces_the_game(self):
        export = self.write_export([stats_row("SJS", 71, "Macklin Celebrini")])
        ingest.import_export(self.con, export, date="2026-10-10")
        ingest.import_export(self.con, export, date="2026-10-11")
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM games").fetchone()[0], 1)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM game_stats").fetchone()[0], 1)

    def test_game_roster_sets_date_teams_and_shared_numbers_need_the_name(self):
        game_roster = [dict(r, as_of="2026-03-03") for r in ROSTER[:3]] + [
            dict(ROSTER[3], as_of="2026-03-03"),
            {**ROSTER[4], "number": 14, "player_id": 7, "player": "Other Fourteen", "as_of": "2026-03-03"}]
        path = self.dir / "2026-03-03_MTL_at_SJS.csv"
        write_roster(path, game_roster)
        export = self.write_export([stats_row("MTL", 14, "Nick Suzuki"), stats_row("MTL", 14, "Somebody Else")])
        _, matched, skipped = ingest.import_export(self.con, export, roster=path)
        self.assertEqual((matched, skipped), (1, 1))
        game = self.con.execute("SELECT * FROM games").fetchone()
        self.assertEqual((game["date"], game["away"], game["home"]), ("2026-03-03", "MTL", "SJS"))
        # The older game roster doesn't move current players, but adds players the database lacked.
        self.assertEqual(self.con.execute("SELECT as_of FROM players WHERE player_id = 1").fetchone()[0], "2026-10-01")
        self.assertIsNotNone(self.con.execute("SELECT 1 FROM players WHERE player_id = 7").fetchone())


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "t.db"
        write_roster(Path(cls.tmp.name) / "roster.csv")
        con = db.connect(path)
        db.load_roster(con, Path(cls.tmp.name) / "roster.csv")
        for i, (shots, turnovers) in enumerate([(4, 0), (5, 0), (3, 1), (6, 0)]):
            rows = [stats_row("SJS", 71, "Macklin Celebrini", shots=shots, shots_on_net=shots, takeaways=2),
                    stats_row("SJS", 48, "Will Smith", turnovers=4 + turnovers),
                    stats_row("MTL", 14, "Nick Suzuki", passes_completed=3)]
            ingest.import_game(con, rows, f"test:{i}", f"2026-10-0{i + 1}", "MTL", "SJS")
        con.close()
        cls.server = server.make_server("127.0.0.1", 0, path, quiet=True)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def test_players_and_teams(self):
        _, teams = self.call("GET", "/api/teams")
        self.assertEqual({t["team"] for t in teams}, {"SJS", "MTL"})
        _, players = self.call("GET", "/api/players?team=SJS&sort=form")
        self.assertEqual([p["name"] for p in players][:2], ["Macklin Celebrini", "Will Smith"])
        self.assertEqual(players[0]["form"]["games"], 4)
        self.assertEqual(players[0]["recommendation"]["action"], "keep")
        self.assertEqual(players[1]["recommendation"]["action"], "trade")
        _, goalies = self.call("GET", "/api/players?group=G")
        self.assertEqual({p["name"] for p in goalies}, {"Yaroslav Askarov", "Sam Montembeault"})

    def test_player_detail_and_game(self):
        _, p = self.call("GET", "/api/players/1")
        self.assertEqual(len(p["games"]), 4)
        self.assertEqual(p["games"][0]["date"], "2026-10-04")  # newest first
        self.assertTrue(p["games"][0]["components"])
        _, g = self.call("GET", f"/api/games/{p['games'][0]['game_id']}")
        self.assertEqual(g["players"][0]["name"], "Macklin Celebrini")
        self.assertEqual(self.call("GET", "/api/players/999")[0], 404)

    def test_favorites(self):
        self.assertEqual(self.call("POST", "/api/players/4/favorite")[0], 201)
        _, favs = self.call("GET", "/api/players?favorites=1")
        self.assertEqual([p["player_id"] for p in favs], [4])
        self.call("DELETE", "/api/players/4/favorite")
        self.assertEqual(self.call("GET", "/api/players?favorites=1")[1], [])

    def test_pool_playlist_and_assessment(self):
        status, pl = self.call("POST", "/api/playlists", {"name": "My pool", "is_pool": True, "slots_f": 2, "slots_d": 0, "slots_g": 0})
        self.assertEqual(status, 201)
        for pid in (1, 2):
            self.call("POST", f"/api/playlists/{pl['id']}/players", {"player_id": pid})
        _, pl = self.call("GET", f"/api/playlists/{pl['id']}")
        self.assertEqual([p["player_id"] for p in pl["players"]], [1, 2])
        _, a = self.call("GET", f"/api/playlists/{pl['id']}/assessment")
        self.assertEqual(a["counts"]["trade"], 1)
        self.assertIn(a["verdict"], ("targeted", "active"))
        _, a2 = self.call("PATCH", f"/api/playlists/{pl['id']}", {"name": "Renamed"})
        self.assertEqual(a2["name"], "Renamed")
        self.call("DELETE", f"/api/playlists/{pl['id']}/players/2")
        self.assertEqual(len(self.call("GET", f"/api/playlists/{pl['id']}")[1]["players"]), 1)
        self.assertEqual(self.call("DELETE", f"/api/playlists/{pl['id']}")[0], 200)
        self.assertEqual(self.call("GET", f"/api/playlists/{pl['id']}")[0], 404)

    def test_bad_requests(self):
        self.assertEqual(self.call("POST", "/api/playlists", {"name": "  "})[0], 400)
        self.assertEqual(self.call("POST", "/api/playlists", {"name": "x", "slots_f": 99})[0], 400)
        self.assertEqual(self.call("PUT", "/api/playlists")[0], 405)
        self.assertEqual(self.call("DELETE", "/api/teams")[0], 405)
        self.assertEqual(self.call("GET", "/api/nope")[0], 404)

    def test_serves_the_page_and_not_outside_static(self):
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn(b"Hockey-Vision", r.read())
        with urllib.request.urlopen(self.base + "/../../webapp/db.py") as r:
            self.assertIn(b"<!doctype html>", r.read())  # falls back to the page


if __name__ == "__main__":
    unittest.main()
