"""The web app's NHL data (webapp/nhl.py): parsing each endpoint's JSON, storing it, and serving it
through the API. The NHL's API is replaced by canned responses shaped like the ones documented at
https://gitlab.com/dword4/nhlapi (new-api.md), so no network is needed."""
import csv
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from webapp import api, db, ingest, nhl  # noqa: E402

TODAY = date(2026, 10, 18)

STANDINGS = {"standings": [
    {"teamAbbrev": {"default": "SJS"}, "teamName": {"default": "San Jose Sharks"}, "conferenceName": "Western",
     "divisionName": "Pacific", "gamesPlayed": 8, "wins": 5, "losses": 2, "otLosses": 1, "points": 11,
     "pointPctg": 0.6875, "goalFor": 27, "goalAgainst": 22, "streakCode": "W", "streakCount": 3,
     "l10Wins": 5, "l10Losses": 2, "l10OtLosses": 1, "divisionSequence": 2, "conferenceSequence": 4, "leagueSequence": 9},
    {"teamAbbrev": {"default": "MTL"}, "teamName": {"default": "Montréal Canadiens"}, "gamesPlayed": 7, "wins": 3,
     "losses": 4, "otLosses": 0, "points": 6},
]}
ROSTERS = {
    "SJS": {"forwards": [{"id": 1, "firstName": {"default": "Macklin"}, "lastName": {"default": "Celebrini"},
                          "sweaterNumber": 71, "positionCode": "C", "shootsCatches": "L", "heightInInches": 72,
                          "weightInPounds": 190, "birthDate": "2006-06-13", "birthCity": {"default": "Vancouver"},
                          "birthStateProvince": {"default": "BC"}, "birthCountry": "CAN",
                          "headshot": "https://assets.nhle.com/mugs/nhl/20262027/SJS/1.png"}],
            "defensemen": [], "goalies": [{"id": 3, "firstName": {"default": "Yaroslav"}, "lastName": {"default": "Askarov"},
                                           "sweaterNumber": 30, "positionCode": "G", "shootsCatches": "L"}]},
    # Nick Suzuki (4) is on the CSV roster as MTL; the API now has him on SJS.
    "MTL": {"forwards": [{"id": 5, "firstName": {"default": "Mike"}, "lastName": {"default": "Matheson"},
                          "sweaterNumber": 8, "positionCode": "D"}], "defensemen": [], "goalies": []},
}
ROSTERS["SJS"]["forwards"].append({"id": 4, "firstName": {"default": "Nick"}, "lastName": {"default": "Suzuki"},
                                   "sweaterNumber": 14, "positionCode": "C"})
SKATERS = {"data": [
    {"playerId": 1, "skaterFullName": "Macklin Celebrini", "teamAbbrevs": "SJS", "gamesPlayed": 8, "goals": 5,
     "assists": 7, "points": 12, "plusMinus": 4, "penaltyMinutes": 2, "ppPoints": 5, "shots": 30,
     "timeOnIcePerGame": 1150.5, "pointsPerGame": 1.5},
    {"playerId": 4, "teamAbbrevs": "MTL,SJS", "gamesPlayed": 7, "goals": 2, "assists": 3, "points": 5},
    {"playerId": 999, "teamAbbrevs": "XXX", "gamesPlayed": 1, "goals": 0, "assists": 0, "points": 0},  # not in the database
]}
GOALIES = {"data": [{"playerId": 3, "teamAbbrevs": "SJS", "gamesPlayed": 6, "gamesStarted": 6, "wins": 4,
                     "losses": 1, "otLosses": 1, "goalsAgainstAverage": 2.31, "savePct": 0.917, "shutouts": 1}]}
SCHEDULE = {"nextStartDate": "2026-10-25", "gameWeek": [
    {"date": "2026-10-18", "games": [{"id": 2026020101, "gameType": 2, "startTimeUTC": "2026-10-18T23:00:00Z",
                                       "gameState": "FUT", "awayTeam": {"abbrev": "MTL"}, "homeTeam": {"abbrev": "SJS"}}]},
    {"date": "2026-10-20", "games": [{"id": 2026020120, "gameType": 2, "gameState": "FUT",
                                       "awayTeam": {"abbrev": "SJS"}, "homeTeam": {"abbrev": "LAK"}}]},
    {"date": "2026-10-26", "games": [{"id": 2026020190, "awayTeam": {"abbrev": "SJS"}, "homeTeam": {"abbrev": "ANA"}}]},
]}
BOXSCORE = {"id": 2025020969, "gameDate": "2026-03-03", "gameState": "OFF",
            "awayTeam": {"abbrev": "MTL", "score": 2, "sog": 28}, "homeTeam": {"abbrev": "SJS", "score": 4, "sog": 33},
            "playerByGameStats": {
                "awayTeam": {"forwards": [{"playerId": 4, "sweaterNumber": 14, "name": {"default": "N. Suzuki"},
                                           "position": "C", "goals": 1, "assists": 1, "points": 2, "plusMinus": -1,
                                           "pim": 0, "sog": 4, "hits": 1, "blockedShots": 0, "toi": "19:42"}],
                             "defense": [], "goalies": [{"playerId": 6, "saveShotsAgainst": "29/33", "goalsAgainst": 4,
                                                         "savePctg": 0.879, "toi": "58:31", "decision": "L"}]},
                "homeTeam": {"forwards": [{"playerId": 1, "goals": 2, "assists": 0, "points": 2, "sog": 5, "toi": "20:05"}],
                             "defense": [], "goalies": []}}}
LANDING = {"playerId": 1, "isActive": True, "currentTeamAbbrev": "SJS", "fullTeamName": {"default": "San Jose Sharks"},
           "sweaterNumber": 71, "headshot": "https://assets.nhle.com/mugs/nhl/20262027/SJS/1.png", "shootsCatches": "L",
           "heightInInches": 72, "weightInPounds": 191, "birthDate": "2006-06-13", "birthCity": {"default": "Vancouver"},
           "birthCountry": "CAN", "draftDetails": {"year": 2024, "teamAbbrev": "SJS", "round": 1, "pickInRound": 1, "overallPick": 1},
           "featuredStats": {"season": 20262027, "regularSeason": {
               "subSeason": {"gamesPlayed": 8, "goals": 5, "assists": 7, "points": 12, "plusMinus": 4},
               "career": {"gamesPlayed": 90, "goals": 30, "assists": 40, "points": 70}}},
           "last5Games": [{"gameId": 2026020090, "gameDate": "2026-10-16", "opponentAbbrev": "VGK", "homeRoadFlag": "H",
                           "goals": 1, "assists": 2, "points": 3, "plusMinus": 2, "shots": 4, "toi": "21:10"}]}


def fake_api(url):
    """The canned response for an NHL API url."""
    if url.endswith("/standings/now"):
        return STANDINGS
    if "/roster/" in url:
        team = url.split("/roster/")[1].split("/")[0]
        if team not in ROSTERS:
            raise nhl.NhlError(f"{url}: HTTP Error 404")
        return ROSTERS[team]
    if "/skater/summary" in url:
        assert "seasonId%3D20262027" in url and "gameTypeId%3D2" in url, url
        return SKATERS
    if "/goalie/summary" in url:
        return GOALIES
    if "/schedule/" in url:
        return SCHEDULE
    if url.endswith("/gamecenter/2025020969/boxscore"):
        return BOXSCORE
    if url.endswith("/player/1/landing"):
        return LANDING
    raise nhl.NhlError(f"{url}: unexpected in tests")


def unreachable(url):
    raise nhl.NhlError(f"{url}: <urlopen error Tunnel connection failed: 403 Forbidden>")


ROSTER_CSV = [
    {"team": "SJS", "team_name": "San Jose Sharks", "number": 71, "player": "Macklin Celebrini", "position": "Center", "player_id": 1, "as_of": "2026-10-01"},
    {"team": "SJS", "team_name": "San Jose Sharks", "number": 30, "player": "Yaroslav Askarov", "position": "Goalie", "player_id": 3, "as_of": "2026-10-01"},
    {"team": "MTL", "team_name": "Montréal Canadiens", "number": 14, "player": "Nick Suzuki", "position": "Center", "player_id": 4, "as_of": "2026-10-01"},
    {"team": "MTL", "team_name": "Montréal Canadiens", "number": 8, "player": "Mike Matheson", "position": "Defense", "player_id": 5, "as_of": "2026-10-01"},
    {"team": "MTL", "team_name": "Montréal Canadiens", "number": 35, "player": "Sam Montembeault", "position": "Goalie", "player_id": 6, "as_of": "2026-10-01"},
]


class ParserTests(unittest.TestCase):
    def test_season_id(self):
        self.assertEqual(nhl.current_season(date(2026, 10, 18)), "20262027")
        self.assertEqual(nhl.current_season(date(2027, 3, 1)), "20262027")
        self.assertEqual(nhl.current_season(date(2027, 9, 2)), "20272028")

    def test_standings(self):
        sjs, mtl = nhl.parse_standings(STANDINGS)
        self.assertEqual((sjs["team"], sjs["team_name"], sjs["points"], sjs["streak"], sjs["last10"]),
                         ("SJS", "San Jose Sharks", 11, "W3", "5-2-1"))
        self.assertEqual((mtl["streak"], mtl["last10"], mtl["division"]), (None, None, None))

    def test_roster(self):
        rows = nhl.parse_roster("SJS", ROSTERS["SJS"])
        first = rows[0]
        self.assertEqual((first["name"], first["position"], first["birth_place"], first["height_in"]),
                         ("Macklin Celebrini", "Center", "Vancouver, BC, CAN", 72))
        self.assertEqual(next(r for r in rows if r["player_id"] == 3)["position"], "Goalie")

    def test_summary(self):
        skaters = nhl.parse_summary(SKATERS, "skater")
        self.assertEqual((skaters[1]["points"], skaters[1]["toi_per_game_s"], skaters[1]["team"]), (12, 1150.5, "SJS"))
        self.assertEqual(skaters[4]["teams"], ["MTL", "SJS"])
        self.assertIsNone(skaters[4]["plus_minus"])  # not sent: blank, not an error
        goalies = nhl.parse_summary(GOALIES, "goalie")
        self.assertEqual((goalies[3]["wins"], goalies[3]["save_pct"], goalies[3]["role"]), (4, 0.917, "goalie"))

    def test_schedule_matches_documented_example_shape(self):
        games = nhl.parse_schedule(SCHEDULE)
        self.assertEqual([(g["date"], g["away"], g["home"]) for g in games],
                         [("2026-10-18", "MTL", "SJS"), ("2026-10-20", "SJS", "LAK"), ("2026-10-26", "SJS", "ANA")])

    def test_boxscore(self):
        game, lines = nhl.parse_boxscore(BOXSCORE)
        self.assertEqual((game["away"], game["away_score"], game["home"], game["home_score"]), ("MTL", 2, "SJS", 4))
        suzuki = next(l for l in lines if l["player_id"] == 4)
        self.assertEqual((suzuki["goals"], suzuki["assists"], suzuki["toi_s"], suzuki["shots"]), (1, 1, 1182, 4))
        goalie = next(l for l in lines if l["player_id"] == 6)
        self.assertEqual((goalie["saves"], goalie["shots_against"], goalie["decision"]), (29, 33, "L"))

    def test_landing(self):
        page = nhl.parse_landing(LANDING)
        self.assertEqual((page["season"]["points"], page["career"]["games"], page["draft"]["overall"]), (12, 90, 1))
        self.assertEqual(page["last5"][0]["opponent"], "VGK")
        self.assertTrue(page["last5"][0]["home"])
        self.assertEqual(nhl.parse_landing({})["last5"], [])


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "t.db"
        roster = Path(self.tmp.name) / "roster.csv"
        with open(roster, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ROSTER_CSV[0]))
            w.writeheader()
            w.writerows(ROSTER_CSV)
        self.con = db.connect(self.path)
        db.load_roster(self.con, roster)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def test_full_sync(self):
        with mock.patch.object(nhl, "fetch_json", fake_api):
            report = nhl.sync(self.con, today=TODAY)
        self.assertTrue(all(r["ok"] for r in report.values()), report)
        player = lambda pid: self.con.execute("SELECT * FROM players WHERE player_id = ?", (pid,)).fetchone()  # noqa: E731
        self.assertEqual((player(4)["team"], player(4)["team_name"], player(4)["as_of"]), ("SJS", "San Jose Sharks", "2026-10-18"))
        self.assertEqual((player(1)["shoots"], player(1)["birth_place"]), ("L", "Vancouver, BC, CAN"))
        # Montembeault (6) is on no roster the API returned: kept, but off the roster and out of MTL's list.
        self.assertEqual([player(pid)["on_roster"] for pid in (1, 4, 5, 6)], [1, 1, 1, 0])
        self.assertIn("1 no longer on a roster", report["rosters"]["detail"])
        self.assertNotIn(6, [p["player_id"] for p in api.players(self.con, {"team": "MTL"}, {})])
        self.assertIn(6, [p["player_id"] for p in api.players(self.con, {"q": "Montembeault"}, {})])
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM nhl_season_stats").fetchone()[0], 3)  # 999 isn't ours
        # Only this week's games are kept from the schedule.
        self.assertEqual([r[0] for r in self.con.execute("SELECT date FROM nhl_schedule ORDER BY date")],
                         ["2026-10-18", "2026-10-20"])
        self.assertTrue(all(r["ok"] for r in self.con.execute("SELECT ok FROM nhl_sync")))

    def test_unreachable_api_is_reported_not_raised(self):
        with mock.patch.object(nhl, "fetch_json", unreachable):
            report = nhl.sync(self.con, today=TODAY)
        self.assertFalse(any(r["ok"] for k, r in report.items() if k != "boxscores"))
        self.assertIn("403", report["standings"]["detail"])
        # Rosters fall back to the database's teams and still report the failure.
        self.assertIn("no rosters fetched", report["rosters"]["detail"])
        self.assertEqual(self.con.execute("SELECT team FROM players WHERE player_id = 4").fetchone()[0], "MTL")

    def test_one_team_failing_keeps_the_rest(self):
        standings = {"standings": STANDINGS["standings"] + [{"teamAbbrev": {"default": "XXX"}, "teamName": {"default": "Nowhere"}}]}
        fetch = lambda url: standings if url.endswith("/standings/now") else fake_api(url)  # noqa: E731
        with mock.patch.object(nhl, "fetch_json", fetch):
            report = nhl.sync(self.con, only=["standings", "rosters"], today=TODAY)
        self.assertTrue(report["rosters"]["ok"])
        self.assertIn("failed: XXX", report["rosters"]["detail"])

    def test_imported_game_gets_its_box_score(self):
        rows = [{"player_id": "SJS #71", "team": "A", "team_abbrev": "SJS", "jersey_number": "71", "role": "skater",
                 "player_name": "Macklin Celebrini", "identified": True, "time_detected_s": 300, "shots": 3}]
        game_id, _, _ = ingest.import_game(self.con, rows, "test:box", "2026-03-03", "MTL", "SJS", nhl_game_id=2025020969)
        with mock.patch.object(nhl, "fetch_json", fake_api):
            report = nhl.sync(self.con, only=["boxscores"])
        self.assertEqual(report["boxscores"]["detail"], "1 games")
        g = api.game(self.con, {}, {}, game_id)
        self.assertEqual((g["official"]["home_score"], g["players"][0]["official"]["goals"]), (4, 2))
        with mock.patch.object(nhl, "fetch_json", unreachable):  # a final box score isn't fetched again
            self.assertEqual(nhl.sync(self.con, only=["boxscores"])["boxscores"]["detail"], "0 games")

    def test_api_serves_nhl_data(self):
        with mock.patch.object(nhl, "fetch_json", fake_api):
            nhl.sync(self.con, today=TODAY)
            page = api.player_nhl(self.con, {}, {}, 1)
        self.assertEqual(page["last5"][0]["points"], 3)
        self.assertEqual(self.con.execute("SELECT weight_lb FROM players WHERE player_id = 1").fetchone()[0], 191)
        with mock.patch.object(nhl, "fetch_json", unreachable):
            self.assertEqual(api.player_nhl(self.con, {}, {}, 1)["fetched_at"], page["fetched_at"])  # cached
            with self.assertRaises(api.ApiError) as e:
                api.player_nhl(self.con, {}, {}, 3)
            self.assertEqual(e.exception.status, 502)

        p = api.player(self.con, {}, {}, 1)
        self.assertEqual((p["season"]["points"], p["standing"]["points"], p["bio"]["shoots"]), (12, 11, "L"))
        by_points = api.players(self.con, {"sort": "points", "team": "SJS"}, {})
        self.assertEqual([x["player_id"] for x in by_points], [1, 4, 3])  # goalies by wins
        sjs = next(t for t in api.teams(self.con, {}, {}) if t["team"] == "SJS")
        self.assertEqual(sjs["standing"]["streak"], "W3")

        pool = api.create_playlist(self.con, {}, {"name": "Pool", "is_pool": True})
        for pid in (1, 3, 5):
            api.add_to_playlist(self.con, {}, {"player_id": pid}, pool["id"])
        with mock.patch.object(api, "date", mock.Mock(today=lambda: TODAY)):
            d = api.dashboard(self.con, {}, {})
        season = d["playlist"]["season"]
        self.assertEqual((season["points"], season["goalie_wins"], season["players"]), (12, 4, 2))
        week = {p["player_id"]: len(p["games"]) for p in d["playlist"]["week"]["players"]}
        self.assertEqual(week, {1: 2, 3: 2, 5: 1})  # SJS play twice this week, MTL once
        self.assertEqual(d["playlist"]["week"]["games"], 5)
        self.assertEqual({k["kind"] for k in d["nhl"]["kinds"]}, set(nhl.KINDS))

    def test_sync_endpoint_validates(self):
        with self.assertRaises(api.ApiError):
            api.nhl_sync(self.con, {}, {"only": ["everything"]})
        with mock.patch.object(nhl, "fetch_json", fake_api):
            out = api.nhl_sync(self.con, {}, {"only": ["standings"]})
        self.assertEqual(list(out["report"]), ["standings"])
        self.assertEqual(out["counts"]["standings"], 2)


if __name__ == "__main__":
    unittest.main()
