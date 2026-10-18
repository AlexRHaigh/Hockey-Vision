"""Up-to-date NHL information for the web app, from the NHL's public APIs as documented at
https://gitlab.com/dword4/nhlapi (new-api.md; the old statsapi.web.nhl.com in stats-api.md was retired
in 2023):

    api-web.nhle.com/v1/standings/now               every team's record           -> nhl_standings
    api-web.nhle.com/v1/roster/<team>/current       rosters, headshots and bios   -> players
    api.nhle.com/stats/rest/en/skater/summary       every skater's season totals  -> nhl_season_stats
    api.nhle.com/stats/rest/en/goalie/summary       every goalie's season totals  -> nhl_season_stats
    api-web.nhle.com/v1/schedule/<date>             the week's games              -> nhl_schedule
    api-web.nhle.com/v1/gamecenter/<id>/boxscore    official lines of imported games -> nhl_boxscores
    api-web.nhle.com/v1/player/<id>/landing         last 5 games, draft, career   -> nhl_player_landing

    python -m webapp.nhl sync                         # all of the above but player pages
    python -m webapp.nhl sync --only stats schedule   # some of it
    python -m webapp.nhl boxscore 2025020969          # one game's official box score

The server's "Refresh NHL data" button runs the same sync, and player pages are fetched (and cached
for PLAYER_PAGE_HOURS) when someone opens a player. Each parser reads only the fields it needs and
leaves anything the API doesn't send as None, so a renamed field shows up as a blank, not a crash.
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path

try:
    from . import db
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from webapp import db

API_WEB = "https://api-web.nhle.com/v1"
API_STATS = "https://api.nhle.com/stats/rest/en"
POSITIONS = {"C": "Center", "L": "Left wing", "R": "Right wing", "D": "Defense", "G": "Goalie"}  # as fetch_roster.py
REGULAR_SEASON = 2
SCHEDULE_DAYS = 7
PLAYER_PAGE_HOURS = 6
KINDS = ("standings", "rosters", "stats", "schedule", "boxscores")


class NhlError(Exception):
    pass


def fetch_json(url, timeout=30):
    # The API turns away Python's default User-Agent.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Hockey-Vision web app)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except (OSError, ValueError) as e:
        raise NhlError(f"{url}: {e}") from e


def _fetch(url):
    """fetch_json, looked up at call time so tests can replace it."""
    return fetch_json(url)


def current_season(today=None):
    """The 8-digit season id (e.g. 20262027): a season starts in September."""
    today = today or Date.today()
    start = today.year if today.month >= 9 else today.year - 1
    return f"{start}{start + 1}"


def _text(v):
    """Names come as {"default": "...", "fr": "..."}."""
    return v.get("default") if isinstance(v, dict) else v


def _seconds(toi):
    """"MM:SS" (or "MMM:SS") -> seconds."""
    try:
        m, s = str(toi).split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return None


def _pick(src, fields):
    return {new: src.get(old) for old, new in fields.items()}


# --- parsers: one per endpoint, NHL JSON in, plain dicts out ---

def parse_standings(data):
    rows = []
    for t in data.get("standings") or []:
        team = _text(t.get("teamAbbrev"))
        if not team:
            continue
        l10 = [t.get(k) for k in ("l10Wins", "l10Losses", "l10OtLosses")]
        rows.append({
            "team": team, "team_name": _text(t.get("teamName")), "conference": t.get("conferenceName"),
            "division": t.get("divisionName"), "games": t.get("gamesPlayed"), "wins": t.get("wins"),
            "losses": t.get("losses"), "ot_losses": t.get("otLosses"), "points": t.get("points"),
            "point_pct": t.get("pointPctg"), "goals_for": t.get("goalFor"), "goals_against": t.get("goalAgainst"),
            "streak": f"{t['streakCode']}{t.get('streakCount', '')}" if t.get("streakCode") else None,
            "last10": "-".join(map(str, l10)) if None not in l10 else None,
            "division_rank": t.get("divisionSequence"), "conference_rank": t.get("conferenceSequence"),
            "league_rank": t.get("leagueSequence"),
        })
    return rows


def parse_roster(team, data):
    rows = []
    for group in ("forwards", "defensemen", "goalies"):
        for p in data.get(group) or []:
            if p.get("id") is None:
                continue
            place = [_text(p.get("birthCity")), _text(p.get("birthStateProvince")), p.get("birthCountry")]
            rows.append({
                "player_id": p["id"], "name": f"{_text(p.get('firstName')) or ''} {_text(p.get('lastName')) or ''}".strip(),
                "team": team, "number": p.get("sweaterNumber"),
                "position": POSITIONS.get(p.get("positionCode"), p.get("positionCode")),
                "headshot": p.get("headshot"), "shoots": p.get("shootsCatches"), "height_in": p.get("heightInInches"),
                "weight_lb": p.get("weightInPounds"), "birth_date": p.get("birthDate"),
                "birth_place": ", ".join(x for x in place if x) or None,
            })
    return rows


SKATER_FIELDS = {"gamesPlayed": "games", "goals": "goals", "assists": "assists", "points": "points",
                 "plusMinus": "plus_minus", "penaltyMinutes": "pim", "ppGoals": "pp_goals", "ppPoints": "pp_points",
                 "shGoals": "sh_goals", "gameWinningGoals": "gw_goals", "shots": "shots", "shootingPct": "shooting_pct",
                 "timeOnIcePerGame": "toi_per_game_s", "faceoffWinPct": "faceoff_pct", "pointsPerGame": "points_per_game"}
GOALIE_FIELDS = {"gamesPlayed": "games", "gamesStarted": "starts", "wins": "wins", "losses": "losses",
                 "otLosses": "ot_losses", "goalsAgainstAverage": "gaa", "savePct": "save_pct", "shutouts": "shutouts",
                 "saves": "saves", "shotsAgainst": "shots_against", "goalsAgainst": "goals_against",
                 "goals": "goals", "assists": "assists", "points": "points"}


def summary_url(role, season):
    expression = urllib.parse.quote(f"seasonId={season} and gameTypeId={REGULAR_SEASON}")
    return f"{API_STATS}/{role}/summary?limit=-1&cayenneExp={expression}"


def parse_summary(data, role):
    """{player_id: season totals} from a stats/rest skater or goalie summary."""
    fields = GOALIE_FIELDS if role == "goalie" else SKATER_FIELDS
    out = {}
    for r in data.get("data") or []:
        if r.get("playerId") is None:
            continue
        teams = [t for t in str(r.get("teamAbbrevs") or "").split(",") if t]
        out[r["playerId"]] = {**_pick(r, fields), "role": role, "team": teams[-1] if teams else None,
                              "teams": teams}
    return out


def parse_schedule(data):
    games = []
    for day in data.get("gameWeek") or []:
        for g in day.get("games") or []:
            away, home = g.get("awayTeam") or {}, g.get("homeTeam") or {}
            if g.get("id") is None:
                continue
            games.append({"nhl_game_id": g["id"], "date": day.get("date"), "start_utc": g.get("startTimeUTC"),
                          "game_type": g.get("gameType"), "state": g.get("gameState"), "venue": _text(g.get("venue")),
                          "away": away.get("abbrev"), "home": home.get("abbrev"),
                          "away_score": away.get("score"), "home_score": home.get("score")})
    return games


def parse_boxscore(data):
    """(game header, [player lines]) from a gamecenter boxscore."""
    away, home = data.get("awayTeam") or {}, data.get("homeTeam") or {}
    game = {"nhl_game_id": data.get("id"), "date": data.get("gameDate"), "state": data.get("gameState"),
            "away": away.get("abbrev"), "home": home.get("abbrev"), "away_score": away.get("score"),
            "home_score": home.get("score"), "away_shots": away.get("sog"), "home_shots": home.get("sog")}
    lines = []
    by_side = data.get("playerByGameStats") or {}
    for side, team in (("awayTeam", game["away"]), ("homeTeam", game["home"])):
        for group in ("forwards", "defense", "goalies"):
            for p in (by_side.get(side) or {}).get(group) or []:
                if p.get("playerId") is None:
                    continue
                line = {"player_id": p["playerId"], "team": team, "name": _text(p.get("name")),
                        "number": p.get("sweaterNumber"), "toi_s": _seconds(p.get("toi")),
                        "role": "goalie" if group == "goalies" else "skater"}
                if group == "goalies":
                    saves, shots = p.get("saves"), p.get("shotsAgainst")
                    if saves is None and "/" in str(p.get("saveShotsAgainst", "")):
                        saves, shots = (int(x) for x in p["saveShotsAgainst"].split("/"))
                    line.update(saves=saves, shots_against=shots, goals_against=p.get("goalsAgainst"),
                                save_pct=p.get("savePctg"), decision=p.get("decision"), starter=p.get("starter"))
                else:
                    line.update(goals=p.get("goals"), assists=p.get("assists"), points=p.get("points"),
                                plus_minus=p.get("plusMinus"), pim=p.get("pim"), shots=p.get("sog"),
                                hits=p.get("hits"), blocks=p.get("blockedShots"), pp_goals=p.get("powerPlayGoals"),
                                giveaways=p.get("giveaways"), takeaways=p.get("takeaways"),
                                faceoff_pct=p.get("faceoffWinningPctg"), shifts=p.get("shifts"))
                lines.append(line)
    return game, lines


LANDING_SEASON = {"gamesPlayed": "games", "goals": "goals", "assists": "assists", "points": "points",
                  "plusMinus": "plus_minus", "pim": "pim", "shots": "shots", "powerPlayGoals": "pp_goals",
                  "powerPlayPoints": "pp_points", "wins": "wins", "losses": "losses", "otLosses": "ot_losses", "goalsAgainstAvg": "gaa",
                  "savePctg": "save_pct", "shutouts": "shutouts"}
LANDING_GAME = {"gameId": "nhl_game_id", "gameDate": "date", "opponentAbbrev": "opponent", "teamAbbrev": "team",
                "goals": "goals", "assists": "assists", "points": "points", "plusMinus": "plus_minus", "pim": "pim",
                "shots": "shots", "powerPlayGoals": "pp_goals", "toi": "toi", "decision": "decision",
                "goalsAgainst": "goals_against", "savePctg": "save_pct", "shotsAgainst": "shots_against"}


def parse_landing(data):
    featured = (data.get("featuredStats") or {}).get("regularSeason") or {}
    draft = data.get("draftDetails") or {}
    return {
        "player_id": data.get("playerId"), "active": data.get("isActive"), "team": data.get("currentTeamAbbrev"),
        "team_name": _text(data.get("fullTeamName")), "number": data.get("sweaterNumber"),
        "headshot": data.get("headshot"), "shoots": data.get("shootsCatches"),
        "height_in": data.get("heightInInches"), "weight_lb": data.get("weightInPounds"),
        "birth_date": data.get("birthDate"),
        "birth_place": ", ".join(x for x in (_text(data.get("birthCity")), _text(data.get("birthStateProvince")),
                                            data.get("birthCountry")) if x) or None,
        "draft": {"year": draft.get("year"), "team": draft.get("teamAbbrev"), "round": draft.get("round"),
                  "pick": draft.get("pickInRound"), "overall": draft.get("overallPick")} if draft else None,
        "season": _pick(featured.get("subSeason") or {}, LANDING_SEASON) if featured.get("subSeason") else None,
        "featured_season": (data.get("featuredStats") or {}).get("season"),
        "career": _pick(featured.get("career") or {}, LANDING_SEASON) if featured.get("career") else None,
        "last5": [{**_pick(g, LANDING_GAME), "home": g.get("homeRoadFlag") == "H"} for g in data.get("last5Games") or []],
    }


# --- sync: fetch, parse and store ---

def _record(con, kind, ok, detail):
    with con:
        con.execute("INSERT INTO nhl_sync (kind, at, ok, detail) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(kind) DO UPDATE SET at = excluded.at, ok = excluded.ok, detail = excluded.detail",
                    (kind, db.now(), int(ok), detail))


def sync_standings(con):
    rows = parse_standings(_fetch(f"{API_WEB}/standings/now"))
    if not rows:
        raise NhlError("standings/now returned no teams")
    with con:
        con.executemany("INSERT OR REPLACE INTO nhl_standings (team, data, updated_at) VALUES (?, ?, ?)",
                        [(r["team"], json.dumps(r), db.now()) for r in rows])
    return f"{len(rows)} teams"


def sync_rosters(con, today=None):
    """Every team's current roster: moves traded players, adds call-ups, fills in bios, and marks
    the players of the fetched teams who are on none of the rosters (sent down, waived, released) as
    off the roster. Teams come from the standings (or the database's players when there are none)."""
    teams = {r["team"]: json.loads(r["data"]).get("team_name") for r in con.execute("SELECT * FROM nhl_standings")}
    if not teams:
        teams = {r["team"]: r["team_name"] for r in con.execute(
            "SELECT DISTINCT team, team_name FROM players WHERE team IS NOT NULL")}
    as_of = (today or Date.today()).isoformat()
    rows, failed, fetched = [], [], set()
    for team, team_name in sorted(teams.items()):
        try:
            rows += [{**r, "team_name": team_name, "as_of": as_of} for r in parse_roster(team, _fetch(f"{API_WEB}/roster/{team}/current"))]
            fetched.add(team)
        except NhlError as e:
            failed.append(f"{team} ({e})")
    if not rows:
        raise NhlError("no rosters fetched: " + "; ".join(failed[:3]))
    with con:
        con.executemany(
            """INSERT INTO players (player_id, name, team, team_name, number, position, as_of,
                                    headshot, shoots, height_in, weight_lb, birth_date, birth_place)
               VALUES (:player_id, :name, :team, :team_name, :number, :position, :as_of,
                       :headshot, :shoots, :height_in, :weight_lb, :birth_date, :birth_place)
               ON CONFLICT(player_id) DO UPDATE SET name = excluded.name, team = excluded.team,
                   team_name = excluded.team_name, number = excluded.number, position = excluded.position,
                   as_of = excluded.as_of, headshot = excluded.headshot, shoots = excluded.shoots,
                   height_in = excluded.height_in, weight_lb = excluded.weight_lb,
                   birth_date = excluded.birth_date, birth_place = excluded.birth_place""", rows)
        on = {r["player_id"] for r in rows}
        con.executemany("UPDATE players SET on_roster = 1 WHERE player_id = ?", [(pid,) for pid in on])
        off = [r[0] for r in con.execute(f"SELECT player_id FROM players WHERE team IN ({', '.join('?' * len(fetched))})",
                                         sorted(fetched)) if r[0] not in on]
        con.executemany("UPDATE players SET on_roster = 0 WHERE player_id = ?", [(pid,) for pid in off])
    return (f"{len(rows)} players on {len(fetched)} teams, {len(off)} no longer on a roster"
            + (f"; failed: {', '.join(failed)}" if failed else ""))


def sync_stats(con, today=None):
    season = current_season(today)
    stats = {}
    for role in ("skater", "goalie"):
        stats.update(parse_summary(_fetch(summary_url(role, season)), role))
    known = {r[0] for r in con.execute("SELECT player_id FROM players")}
    rows = [(pid, season, s["role"], json.dumps(s), db.now()) for pid, s in stats.items() if pid in known]
    with con:
        con.execute("DELETE FROM nhl_season_stats WHERE season != ?", (season,))
        con.executemany("INSERT OR REPLACE INTO nhl_season_stats (player_id, season, role, data, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)", rows)
    return f"{len(rows)} players, season {season[:4]}-{season[6:]}"


def sync_schedule(con, today=None, days=SCHEDULE_DAYS):
    """The games from `today` for `days` days (schedule/<date> returns a week from that date)."""
    start = today or Date.today()
    end = start + timedelta(days=days)
    games, day = [], start
    while day < end:
        week = parse_schedule(_fetch(f"{API_WEB}/schedule/{day.isoformat()}"))
        games += [g for g in week if g["date"] and start.isoformat() <= g["date"] < end.isoformat()]
        day += timedelta(days=7)
    with con:
        con.execute("DELETE FROM nhl_schedule WHERE date < ?", ((start - timedelta(days=30)).isoformat(),))
        con.executemany("INSERT OR REPLACE INTO nhl_schedule (nhl_game_id, date, away, home, data, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        [(g["nhl_game_id"], g["date"], g["away"], g["home"], json.dumps(g), db.now()) for g in games])
    return f"{len(games)} games from {start.isoformat()}"


def sync_boxscore(con, nhl_game_id):
    game, lines = parse_boxscore(_fetch(f"{API_WEB}/gamecenter/{int(nhl_game_id)}/boxscore"))
    with con:
        con.execute("INSERT OR REPLACE INTO nhl_boxscores (nhl_game_id, data, updated_at) VALUES (?, ?, ?)",
                    (nhl_game_id, json.dumps(game), db.now()))
        con.execute("DELETE FROM nhl_boxscore_lines WHERE nhl_game_id = ?", (nhl_game_id,))
        con.executemany("INSERT INTO nhl_boxscore_lines (nhl_game_id, player_id, team, data) VALUES (?, ?, ?, ?)",
                        [(nhl_game_id, l["player_id"], l["team"], json.dumps(l)) for l in lines])
    return game, lines


def sync_boxscores(con):
    """Box scores of imported games with an NHL game id that don't have a final one yet."""
    have = {r["nhl_game_id"]: json.loads(r["data"]).get("state") for r in con.execute("SELECT * FROM nhl_boxscores")}
    ids = [r[0] for r in con.execute("SELECT DISTINCT nhl_game_id FROM games WHERE nhl_game_id IS NOT NULL")
           if have.get(r[0]) not in ("FINAL", "OFF")]
    failed = []
    for game_id in ids:
        try:
            sync_boxscore(con, game_id)
        except NhlError as e:
            failed.append(f"{game_id} ({e})")
    return f"{len(ids) - len(failed)} games" + (f"; failed: {', '.join(failed)}" if failed else "")


def player_landing(con, player_id, max_age_hours=PLAYER_PAGE_HOURS):
    """A player's NHL page, from the cache when it's fresh; keeps the player's bio up to date."""
    row = con.execute("SELECT * FROM nhl_player_landing WHERE player_id = ?", (player_id,)).fetchone()
    if row and datetime.fromisoformat(row["fetched_at"]) > datetime.now(timezone.utc) - timedelta(hours=max_age_hours):
        return {**json.loads(row["data"]), "fetched_at": row["fetched_at"]}
    page = parse_landing(_fetch(f"{API_WEB}/player/{int(player_id)}/landing"))
    fetched_at = db.now()
    with con:
        con.execute("INSERT OR REPLACE INTO nhl_player_landing (player_id, data, fetched_at) VALUES (?, ?, ?)",
                    (player_id, json.dumps(page), fetched_at))
        bio = {k: page[k] for k in db.PLAYER_BIO if page.get(k) is not None}
        if bio:
            con.execute(f"UPDATE players SET {', '.join(f'{k} = ?' for k in bio)} WHERE player_id = ?",
                        [*bio.values(), player_id])
    return {**page, "fetched_at": fetched_at}


def sync(con, only=None, today=None):
    """Refresh each kind of NHL data in turn; one failing doesn't stop the rest.
    Returns {kind: {"ok": bool, "detail": str}}."""
    steps = {"standings": lambda: sync_standings(con), "rosters": lambda: sync_rosters(con, today),
             "stats": lambda: sync_stats(con, today), "schedule": lambda: sync_schedule(con, today),
             "boxscores": lambda: sync_boxscores(con)}
    report = {}
    for kind in KINDS:
        if only and kind not in only:
            continue
        try:
            report[kind] = {"ok": True, "detail": steps[kind]()}
        except NhlError as e:
            report[kind] = {"ok": False, "detail": str(e)}
        _record(con, kind, report[kind]["ok"], report[kind]["detail"])
    return report


def main():
    parser = argparse.ArgumentParser(description="Refresh the web app's NHL data from the NHL's API.")
    parser.add_argument("--db", type=Path, default=db.DEFAULT_DB, help="Database file (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_sync = sub.add_parser("sync", help="Standings, rosters, season stats, this week's schedule, box scores")
    p_sync.add_argument("--only", nargs="+", choices=KINDS, help="Refresh only these")
    p_box = sub.add_parser("boxscore", help="One game's official box score")
    p_box.add_argument("game_id", type=int)
    args = parser.parse_args()

    con = db.connect(args.db)
    db.ensure_players(con)
    if args.command == "boxscore":
        try:
            game, lines = sync_boxscore(con, args.game_id)
        except NhlError as e:
            sys.exit(str(e))
        print(f"Game {args.game_id}: {game['away']} {game['away_score']} at {game['home']} {game['home_score']} "
              f"({game['state']}), {len(lines)} player lines")
        return
    report = sync(con, args.only)
    for kind, r in report.items():
        print(f"  {kind}: {'ok' if r['ok'] else 'FAILED'}: {r['detail']}")
    if not any(r["ok"] for r in report.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
