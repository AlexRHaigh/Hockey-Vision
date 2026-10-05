"""The two teams' rosters for a video: which jersey numbers each detected player can wear, and who
wears them. Built by run_models.py from --teams (team_a and team_b's abbreviations, e.g. SJS MTL)
and --roster (a CSV from fetch_roster.py).

Which team is team_a and which team_b depends on the video: the player model's team_a / goalie_a
classes are one team's players, team_b / goalie_b the other's. Whatever works that out (e.g. the
broadcast's score bug) passes the two abbreviations in that order.

Per player model class:
    team_a_player / team_b_player   that team's skaters' numbers
    goalie_a / goalie_b             that team's goalies' numbers
    referee                         no numbers: referees aren't on rosters, so they get none
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ROSTER = Path(__file__).parent / "rosters" / "nhl_active_players.csv"
CLASS_TEAM = {"team_a_player": (0, "skater"), "goalie_a": (0, "goalie"),
              "team_b_player": (1, "skater"), "goalie_b": (1, "goalie")}


class Roster:
    def __init__(self, path, teams):
        """`teams`: (team_a abbreviation, team_b abbreviation)."""
        path = Path(path)
        if not path.exists():
            sys.exit(f"Roster not found: {path} (fetch it with: python fetch_roster.py)")
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        self.teams = tuple(t.upper() for t in teams)
        available = sorted({r["team"] for r in rows})
        for t in self.teams:
            if t not in available:
                sys.exit(f"No team {t} in {path}; it has: {', '.join(available)}")
        # (team index, role) -> {number: player name}; numbers two players share are both named.
        self.names = defaultdict(dict)
        self.team_names = {}
        for r in rows:
            if r["team"] not in self.teams:
                continue
            i = self.teams.index(r["team"])
            role = "goalie" if r["position"] == "Goalie" else "skater"
            num = str(int(r["number"]))
            names = self.names[(i, role)]
            names[num] = f"{names[num]} / {r['player']}" if num in names else r["player"]
            self.team_names[r["team"]] = r["team_name"]
        self.path = path

    def allowed(self, cls):
        """The numbers a player of model class `cls` can wear: a list, empty for referees, or
        None for a class this doesn't know (no limit)."""
        if cls == "referee":
            return []
        key = CLASS_TEAM.get(cls)
        return None if key is None else sorted(self.names[key], key=int)

    def player(self, cls, number):
        """(team abbreviation, player name) for a player of class `cls` wearing `number`: the team
        from the class alone (None for referees), the name only when the number is known."""
        key = CLASS_TEAM.get(cls)
        if key is None:
            return None, None
        return self.teams[key[0]], None if number is None else self.names[key].get(str(number))

    def describe(self):
        return ", ".join(f"team_{'ab'[i]} = {t} ({self.team_names[t]}, "
                         f"{len(self.names[(i, 'skater')])} skaters, {len(self.names[(i, 'goalie')])} goalies)"
                         for i, t in enumerate(self.teams))
