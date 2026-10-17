"""Fill the web app with made-up games, to try it before real games have been run through the
pipeline. Demo games are labelled "Demo" everywhere in the app and can be removed in one go.

    python -m webapp.demo                 # 300 demo games (about 7 weeks) between the real NHL rosters
    python -m webapp.demo --games 40 --seed 7
    python -m webapp.demo --clear         # remove every demo game

Each player gets a hidden skill level, and some get better or worse over the schedule, so form,
trends and trade calls have something to find. The stats look like player_stats.csv rows and are
imported through the same path as real exports (ingest.import_game).
"""

import argparse
import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    from . import db, ingest
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from webapp import db, ingest

DRESSED = {"F": 12, "D": 6, "G": 1}
# Events per minute detected for an average skater.
SKATER_RATES = {"shots": 0.35, "passes_completed": 0.6, "takeaways": 0.08, "interceptions": 0.1,
                "defensive_plays": 0.2, "blocked_shots": 0.05, "turnovers": 0.12, "times_defended": 0.15}


def poisson(rng, lam):
    """A Poisson draw (Knuth's method; the rates here are small)."""
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def skater_row(rng, player, skill):
    minutes = rng.uniform(2.0, 9.0)
    s = {k: poisson(rng, rate * minutes * (skill if k not in ("turnovers", "times_defended") else 1 / skill))
         for k, rate in SKATER_RATES.items()}
    if player["position"] == "Defense":
        s["blocked_shots"] += poisson(rng, 0.1 * minutes)
        s["shots"] = poisson(rng, 0.2 * minutes * skill)
    s["shots_on_net"] = sum(rng.random() < 0.55 for _ in range(s["shots"]))
    s["possible_goals"] = sum(rng.random() < 0.12 * skill for _ in range(s["shots_on_net"]))
    s["shots_blocked_by_opponent"] = s["shots"] - s["shots_on_net"] - poisson(rng, 0.1)
    s["shots_blocked_by_opponent"] = max(0, s["shots_blocked_by_opponent"])
    s["good_passes"] = sum(rng.random() < 0.3 * skill for _ in range(s["passes_completed"]))
    s["passes_received"] = poisson(rng, 0.55 * minutes)
    s["possessions"] = s["passes_completed"] + s["shots"] + s["turnovers"]
    s["possession_time_s"] = round(s["possessions"] * rng.uniform(1.5, 4.0), 2)
    return minutes, s


def goalie_row(rng, skill):
    minutes = rng.uniform(10.0, 25.0)
    return minutes, {"saves": poisson(rng, 0.9 * minutes * skill), "goalie_recoveries": poisson(rng, 0.5 * minutes),
                     "passes_completed": poisson(rng, 0.1 * minutes), "turnovers": poisson(rng, 0.03 * minutes / skill)}


def team_rows(rng, players, profile, progress):
    groups = {"F": [], "D": [], "G": []}
    for p in players:
        groups[{"Defense": "D", "Goalie": "G"}.get(p["position"], "F")].append(p)
    rows = []
    for g, n in DRESSED.items():
        for p in rng.sample(groups[g], min(n, len(groups[g]))):
            base, drift = profile[p["player_id"]]
            skill = max(0.3, base + drift * progress + rng.gauss(0, 0.25))
            minutes, stats = goalie_row(rng, skill) if g == "G" else skater_row(rng, p, skill)
            rows.append({"player_id": f"{p['team']} #{p['number']}", "team": "", "team_abbrev": p["team"],
                         "jersey_number": str(p["number"]), "player_name": p["name"],
                         "role": "goalie" if g == "G" else "skater", "identified": True,
                         "time_detected_s": round(minutes * 60, 2), **stats})
    return rows


def generate(con, n_games, seed, end=None):
    rng = random.Random(seed)
    players = [dict(r) for r in con.execute("SELECT * FROM players WHERE team IS NOT NULL")]
    by_team = {}
    for p in players:
        by_team.setdefault(p["team"], []).append(p)
    teams = sorted(by_team)
    if len(teams) < 2:
        sys.exit("No rosters in the database (fetch them with: python fetch_roster.py)")
    # Hidden skill (1 = average) and how much it changes from the first demo game to the last.
    profile = {p["player_id"]: (rng.lognormvariate(0, 0.3), rng.choice([0, 0, 0, 0.5, -0.5]) * rng.random())
               for p in players}
    end = end or date.today()
    start = end - timedelta(days=max(1, n_games // 6))
    for i in range(n_games):
        away, home = rng.sample(teams, 2)
        day = start + timedelta(days=round(i * (end - start).days / max(1, n_games - 1)))
        progress = i / max(1, n_games - 1)
        rows = team_rows(rng, by_team[away], profile, progress) + team_rows(rng, by_team[home], profile, progress)
        ingest.import_game(con, rows, f"demo:{seed}:{i}", day.isoformat(), away, home,
                           f"Demo: {away} at {home}", is_demo=True)


def clear(con):
    with con:
        return con.execute("DELETE FROM games WHERE is_demo = 1").rowcount


def main():
    parser = argparse.ArgumentParser(description="Add made-up demo games to the Hockey-Vision web app.")
    parser.add_argument("--db", type=Path, default=db.DEFAULT_DB, help="Database file (default: %(default)s)")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clear", action="store_true", help="Remove every demo game instead")
    args = parser.parse_args()
    con = db.connect(args.db)
    db.ensure_players(con)
    if args.clear:
        print(f"Removed {clear(con)} demo games")
        return
    clear(con)
    generate(con, args.games, args.seed)
    print(f"Added {args.games} demo games to {args.db}")


if __name__ == "__main__":
    main()
