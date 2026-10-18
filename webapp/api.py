"""The web app's JSON API, as plain functions of a database connection. server.py routes HTTP
requests to them (ROUTES); tests call them directly.

Scores depend on every game in the database (scoring.calibrate), so they are computed together and
cached until games are imported or removed. Official NHL data (season totals, standings, schedule,
box scores) comes from the tables nhl.py fills, so pages never wait on the NHL's API, except a
player's NHL page (player_nhl), which is fetched when first opened.
"""

import json
import threading
from datetime import date, timedelta
from statistics import mean, median

from . import db, nhl, scoring


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


_cache, _cache_lock = {}, threading.Lock()


def scored(con):
    """{"by_player": {player_id: [game entries, newest first]}, "by_game": {...}, "forms": {...},
    "calibration": {...}}, cached per database and data version."""
    key = (con.execute("PRAGMA database_list").fetchone()["file"], db.data_version(con))
    with _cache_lock:
        if _cache.get("key") == key:
            return _cache["value"]
    rows = db.all_player_games(con)
    calibration = scoring.calibrate([("goalie" if r["role"] == "goalie" else "skater",
                                      scoring.calibration_inputs(r["stats"], r["role"], r["line"])) for r in rows])
    by_player, by_game = {}, {}
    for r in rows:
        entry = {"game_id": r["game_id"], "date": r["date"], "label": r["label"], "away": r["away"], "home": r["home"],
                 "is_demo": bool(r["is_demo"]), "team": r["team"], "role": r["role"], "player_id": r["player_id"],
                 **scoring.game_score(r["stats"], r["role"], calibration, r["line"]), "stats": r["stats"] or {},
                 "official": r["line"]}
        by_player.setdefault(r["player_id"], []).append(entry)
        by_game.setdefault(r["game_id"], []).append(entry)
    value = {"by_player": by_player, "by_game": by_game, "calibration": calibration,
             "forms": {pid: scoring.form(games) for pid, games in by_player.items()}}
    with _cache_lock:
        _cache.update(key=key, value=value)
    return value


def _player(row, data, favorites=frozenset(), seasons=None):
    f = data["forms"].get(row["player_id"]) or scoring.form([])
    return {"player_id": row["player_id"], "name": row["name"], "team": row["team"], "team_name": row["team_name"],
            "number": row["number"], "position": row["position"], "group": scoring.position_group(row["position"]),
            "headshot": row["headshot"], "on_roster": row["on_roster"], "favorite": row["player_id"] in favorites, "form": f,
            "recommendation": scoring.recommendation(f), "season": (seasons or {}).get(row["player_id"])}


def _seasons(con):
    """{player_id: this season's official NHL totals}."""
    return {r["player_id"]: json.loads(r["data"]) for r in con.execute("SELECT player_id, data FROM nhl_season_stats")}


def _standings(con):
    return {r["team"]: json.loads(r["data"]) for r in con.execute("SELECT team, data FROM nhl_standings")}


def _favorites(con):
    return {r[0] for r in con.execute("SELECT player_id FROM favorites")}


def _get_player_row(con, player_id):
    row = con.execute("SELECT * FROM players WHERE player_id = ?", (player_id,)).fetchone()
    if not row:
        raise ApiError(404, f"No player {player_id}")
    return row


# --- players and teams ---

def teams(con, params, body):
    standings = _standings(con)
    return [{**dict(r), "standing": standings.get(r["team"])} for r in con.execute(
        """SELECT team, team_name, COUNT(*) AS players FROM players
           WHERE team IS NOT NULL AND COALESCE(on_roster, 1) = 1
           GROUP BY team, team_name ORDER BY team_name""")]


def players(con, params, body):
    """Players filtered by ?team=, ?q= (name), ?group= (F/D/G), ?favorites=1, ?rated=1;
    ?sort=form|points|name|number (points: this season's NHL points, goalies by wins)."""
    sql, args = "SELECT * FROM players WHERE 1", []
    if params.get("team"):  # a team's current roster; searching all teams also finds players off one
        sql += " AND team = ? AND COALESCE(on_roster, 1) = 1"
        args.append(params["team"].upper())
    if params.get("q"):
        sql += " AND name LIKE ?"
        args.append(f"%{params['q']}%")
    if params.get("favorites") == "1":
        sql += " AND player_id IN (SELECT player_id FROM favorites)"
    data, favs, seasons = scored(con), _favorites(con), _seasons(con)
    out = [_player(r, data, favs, seasons) for r in con.execute(sql + " ORDER BY name", args)]
    if params.get("group"):
        out = [p for p in out if p["group"] == params["group"].upper()]
    if params.get("rated") == "1":
        out = [p for p in out if p["form"]["games"]]
    sort = params.get("sort", "name")
    if sort == "form":
        out.sort(key=lambda p: (p["form"]["form"] is None, -(p["form"]["form"] or 0)))
    elif sort == "points":
        out.sort(key=lambda p: (p["season"] is None, -_season_points(p["season"])))
    elif sort == "number":
        out.sort(key=lambda p: (p["group"] != "F", p["group"] == "G", p["number"] or 0))
    limit = int(params.get("limit", 0) or 0)
    return out[:limit] if limit else out


def _season_points(season):
    if not season:
        return 0
    return (season.get("wins") or 0) if season.get("role") == "goalie" else (season.get("points") or 0)


def player_nhl(con, params, body, player_id):
    """The player's NHL page: last 5 games, draft, career. Fetched from the NHL on first view, then cached."""
    _get_player_row(con, player_id)
    try:
        return nhl.player_landing(con, player_id)
    except nhl.NhlError as e:
        raise ApiError(502, f"Couldn't reach the NHL's API: {e}")


def player(con, params, body, player_id):
    row = _get_player_row(con, player_id)
    data = scored(con)
    p = _player(row, data, _favorites(con), _seasons(con))
    p["bio"] = {k: row[k] for k in db.PLAYER_BIO}
    p["standing"] = _standings(con).get(row["team"])
    p["games"] = data["by_player"].get(player_id, [])
    p["playlists"] = [dict(r) for r in con.execute(
        """SELECT l.id, l.name, l.is_pool FROM playlists l JOIN playlist_players pp ON pp.playlist_id = l.id
           WHERE pp.player_id = ? ORDER BY l.name""", (player_id,))]
    p["calibration"] = data["calibration"]
    return p


# --- games ---

def games(con, params, body):
    data = scored(con)
    out = []
    for r in con.execute("SELECT * FROM games ORDER BY date DESC, id DESC"):
        entries = data["by_game"].get(r["id"], [])
        top = max(entries, key=lambda e: e["score"], default=None)
        out.append({**dict(r), "is_demo": bool(r["is_demo"]), "players": len(entries),
                    "top": {"player_id": top["player_id"], "score": top["score"]} if top else None})
    names = {r["player_id"]: r["name"] for r in con.execute("SELECT player_id, name FROM players")}
    for g in out:
        if g["top"]:
            g["top"]["name"] = names.get(g["top"]["player_id"])
    return out


def game(con, params, body, game_id):
    row = con.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    if not row:
        raise ApiError(404, f"No game {game_id}")
    names = {r["player_id"]: dict(r) for r in con.execute("SELECT player_id, name, number, position FROM players")}
    entries = sorted(scored(con)["by_game"].get(game_id, []), key=lambda e: -e["score"])
    official = None
    if row["nhl_game_id"]:
        box = con.execute("SELECT data FROM nhl_boxscores WHERE nhl_game_id = ?", (row["nhl_game_id"],)).fetchone()
        official = json.loads(box["data"]) if box else None
    return {**dict(row), "is_demo": bool(row["is_demo"]), "official": official,
            "players": [{**e, **names.get(e["player_id"], {})} for e in entries]}


def delete_game(con, params, body, game_id):
    with con:
        con.execute("DELETE FROM games WHERE id = ?", (game_id,))
    return {"deleted": game_id}


# --- favourites ---

def add_favorite(con, params, body, player_id):
    _get_player_row(con, player_id)
    with con:
        con.execute("INSERT OR IGNORE INTO favorites (player_id, added_at) VALUES (?, ?)", (player_id, db.now()))
    return {"player_id": player_id, "favorite": True}


def remove_favorite(con, params, body, player_id):
    with con:
        con.execute("DELETE FROM favorites WHERE player_id = ?", (player_id,))
    return {"player_id": player_id, "favorite": False}


# --- playlists and pools ---

PLAYLIST_FIELDS = {"name": str, "description": str, "is_pool": bool, "slots_f": int, "slots_d": int, "slots_g": int}


def _playlist_values(body, partial):
    values = {}
    for k, kind in PLAYLIST_FIELDS.items():
        if k in body:
            try:
                values[k] = kind(body[k]) if kind is not bool else bool(body[k])
            except (TypeError, ValueError):
                raise ApiError(400, f"Bad {k}")
            if kind is int and not 0 <= values[k] <= 50:
                raise ApiError(400, f"{k} must be 0-50")
    if "name" in values:
        values["name"] = values["name"].strip()
        if not values["name"]:
            raise ApiError(400, "A playlist needs a name")
    elif not partial:
        raise ApiError(400, "A playlist needs a name")
    return values


def _get_playlist_row(con, playlist_id):
    row = con.execute("SELECT * FROM playlists WHERE id = ?", (playlist_id,)).fetchone()
    if not row:
        raise ApiError(404, f"No playlist {playlist_id}")
    return row


def _playlist_out(row):
    return {**dict(row), "is_pool": bool(row["is_pool"])}


def playlists(con, params, body):
    return [{**_playlist_out(r), "players": r["n"]} for r in con.execute(
        """SELECT l.*, COUNT(pp.player_id) AS n FROM playlists l LEFT JOIN playlist_players pp ON pp.playlist_id = l.id
           GROUP BY l.id ORDER BY l.is_pool DESC, l.name""")]


def create_playlist(con, params, body):
    values = _playlist_values(body, partial=False)
    with con:
        cols = list(values) + ["created_at"]
        pid = con.execute(f"INSERT INTO playlists ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                          [*values.values(), db.now()]).lastrowid
    return playlist(con, params, body, pid)


def update_playlist(con, params, body, playlist_id):
    _get_playlist_row(con, playlist_id)
    values = _playlist_values(body, partial=True)
    if values:
        with con:
            con.execute(f"UPDATE playlists SET {', '.join(f'{k} = ?' for k in values)} WHERE id = ?",
                        [*values.values(), playlist_id])
    return playlist(con, params, body, playlist_id)


def delete_playlist(con, params, body, playlist_id):
    _get_playlist_row(con, playlist_id)
    with con:
        con.execute("DELETE FROM playlists WHERE id = ?", (playlist_id,))
    return {"deleted": playlist_id}


def _members(con, playlist_id):
    return [r["player_id"] for r in con.execute(
        "SELECT player_id FROM playlist_players WHERE playlist_id = ? ORDER BY added_at", (playlist_id,))]


def playlist(con, params, body, playlist_id):
    row = _get_playlist_row(con, playlist_id)
    data, favs, seasons = scored(con), _favorites(con), _seasons(con)
    ids = _members(con, playlist_id)
    rows = {r["player_id"]: r for r in con.execute(
        f"SELECT * FROM players WHERE player_id IN ({', '.join('?' * len(ids))})", ids)} if ids else {}
    members = [_player(rows[i], data, favs, seasons) for i in ids if i in rows]
    members.sort(key=lambda p: ("FDG".index(p["group"]), p["form"]["form"] is None, -(p["form"]["form"] or 0)))
    return {**_playlist_out(row), "players": members}


def add_to_playlist(con, params, body, playlist_id):
    _get_playlist_row(con, playlist_id)
    try:
        player_id = int(body.get("player_id"))
    except (TypeError, ValueError):
        raise ApiError(400, "player_id is required")
    _get_player_row(con, player_id)
    with con:
        con.execute("INSERT OR IGNORE INTO playlist_players (playlist_id, player_id, added_at) VALUES (?, ?, ?)",
                    (playlist_id, player_id, db.now()))
    return {"playlist_id": playlist_id, "player_id": player_id}


def remove_from_playlist(con, params, body, playlist_id, player_id):
    with con:
        con.execute("DELETE FROM playlist_players WHERE playlist_id = ? AND player_id = ?", (playlist_id, player_id))
    return {"playlist_id": playlist_id, "player_id": player_id, "removed": True}


def assessment(con, params, body, playlist_id):
    """How strong the pool is, who to trade, who to trade for, and whether to trade at all."""
    row = _get_playlist_row(con, playlist_id)
    data = scored(con)
    members = set(_members(con, playlist_id))
    pool, others = [], []
    for r in con.execute("SELECT * FROM players"):
        p = _player(r, data)
        if r["player_id"] in members:
            pool.append(p)
        elif p["form"]["games"] and r["on_roster"] != 0:  # no upgrades to players off every roster
            others.append(p)
    result = scoring.assess_pool(pool, others, {"F": row["slots_f"], "D": row["slots_d"], "G": row["slots_g"]})
    return {"playlist_id": playlist_id, **result}


# --- overview ---

OVERVIEW_WEEKS = 8
OVERVIEW_GAMES = 6
OVERVIEW_PLAYERS = 5


def _week(day):
    d = date.fromisoformat(day)
    return (d - timedelta(days=d.weekday())).isoformat()


def _mean(values):
    values = [v for v in values if v is not None]
    return round(mean(values), 1) if values else None


def _season_totals(members):
    """Your players' official NHL totals this season, or None before the stats are synced."""
    seasons = [p["season"] for p in members if p["season"]]
    if not seasons:
        return None
    skaters = [x for x in seasons if x.get("role") != "goalie"]
    goalies = [x for x in seasons if x.get("role") == "goalie"]
    total = lambda rows, k: sum(x.get(k) or 0 for x in rows)  # noqa: E731
    games, points = total(skaters, "games"), total(skaters, "points")
    return {"players": len(seasons), "skaters": len(skaters), "goalies": len(goalies), "games": games,
            "goals": total(skaters, "goals"), "assists": total(skaters, "assists"), "points": points,
            "points_per_game": round(points / games, 2) if games else None,
            "goalie_wins": total(goalies, "wins"), "shutouts": total(goalies, "shutouts")}


def _week_schedule(con, today=None):
    """{team: [their games in the next 7 days]} from the synced schedule, or None if there is none."""
    start = today or date.today()
    end = start + timedelta(days=nhl.SCHEDULE_DAYS)
    rows = con.execute("SELECT data FROM nhl_schedule WHERE date >= ? AND date < ? ORDER BY date",
                       (start.isoformat(), end.isoformat())).fetchall()
    if not rows:
        return None
    out = {}
    for r in rows:
        g = json.loads(r["data"])
        for team, opponent, home in ((g["away"], g["home"], False), (g["home"], g["away"], True)):
            out.setdefault(team, []).append({"date": g["date"], "opponent": opponent, "home": home,
                                             "start_utc": g.get("start_utc")})
    return {"start": start.isoformat(), "end": (end - timedelta(days=1)).isoformat(), "teams": out}


def _setup_stats(data, members, league_medians, schedule=None):
    """Statistics for a group of your players: form, calls, weekly scores, position groups, their
    recent games, the best of them and the ones needing attention; with NHL data synced, their
    official season totals and their games this week."""
    rated = [p for p in members if p["form"]["games"]]
    games = [g for p in members for g in data["by_player"].get(p["player_id"], [])]
    names = {p["player_id"]: p["name"] for p in members}

    weekly = {}
    for g in games:
        weekly.setdefault(_week(g["date"]), []).append(g["score"])
    by_game = {}
    for g in games:
        by_game.setdefault(g["game_id"], []).append(g)
    recent = sorted(by_game.values(), key=lambda gs: (gs[0]["date"], gs[0]["game_id"]), reverse=True)[:OVERVIEW_GAMES]

    groups = []
    for key, name in scoring.GROUP_NAMES.items():
        forms = [p["form"]["form"] for p in rated if p["group"] == key]
        groups.append({"group": key, "name": name, "players": sum(p["group"] == key for p in members),
                       "rated": len(forms), "form": _mean(forms), "league_median": league_medians.get(key)})
    by_form = sorted(rated, key=lambda p: -p["form"]["form"])
    return {
        "size": len(members), "rated": len(rated), "games": len(games),
        "form": _mean(p["form"]["form"] for p in rated), "last": _mean(p["form"]["last"] for p in rated),
        "trend": _mean(p["form"]["trend"] for p in rated),
        "counts": {a: sum(p["recommendation"]["action"] == a for p in members) for a in ("keep", "watch", "trade", "hold")},
        "weekly": [{"week": w, "score": round(mean(v), 1), "games": len(v)} for w, v in sorted(weekly.items())[-OVERVIEW_WEEKS:]],
        "groups": groups,
        "recent": [{"game_id": gs[0]["game_id"], "date": gs[0]["date"], "label": gs[0]["label"], "is_demo": gs[0]["is_demo"],
                    "score": _mean(g["score"] for g in gs),
                    "players": [{"player_id": g["player_id"], "name": names[g["player_id"]], "score": g["score"]}
                                for g in sorted(gs, key=lambda g: -g["score"])]} for gs in recent],
        "season": _season_totals(members),
        "week": _week_games(members, schedule),
        "top": by_form[:OVERVIEW_PLAYERS],
        "attention": sorted((p for p in rated if p["recommendation"]["action"] in ("trade", "watch")),
                            key=lambda p: p["form"]["form"])[:OVERVIEW_PLAYERS],
    }


def _week_games(members, schedule):
    if not schedule:
        return None
    players = [{"player_id": p["player_id"], "name": p["name"], "team": p["team"], "position": p["position"],
                "games": schedule["teams"].get(p["team"], [])} for p in members]
    players.sort(key=lambda p: (-len(p["games"]), p["name"]))
    return {"start": schedule["start"], "end": schedule["end"], "games": sum(len(p["games"]) for p in players),
            "players": players}


def dashboard(con, params, body):
    """The overview of your own setup: one of your playlists (?playlist=, else your first pool) and
    your favourites, measured against the league."""
    lists = playlists(con, {}, {})
    try:
        chosen = int(params["playlist"]) if params.get("playlist") else (lists[0]["id"] if lists else None)
    except ValueError:
        raise ApiError(400, "playlist must be a playlist id")
    data, favs, seasons = scored(con), _favorites(con), _seasons(con)
    schedule = _week_schedule(con)
    everyone = [_player(r, data, favs, seasons) for r in con.execute("SELECT * FROM players")]
    league_medians = {}
    for key in scoring.GROUP_NAMES:
        forms = [p["form"]["form"] for p in everyone if p["group"] == key and p["form"]["games"]]
        league_medians[key] = round(median(forms), 1) if forms else None

    out = {"playlists": lists, "playlist": None, "assessment": None,
           "favorites": _setup_stats(data, [p for p in everyone if p["favorite"]], league_medians, schedule),
           "nhl": nhl_status(con, {}, {}),
           "demo_games": con.execute("SELECT COUNT(*) FROM games WHERE is_demo").fetchone()[0],
           "thresholds": {"trade_below": scoring.TRADE_BELOW, "watch_below": scoring.WATCH_BELOW,
                          "core_above": scoring.CORE_ABOVE, "min_games": scoring.MIN_GAMES}}
    if chosen is not None:
        row = _get_playlist_row(con, chosen)
        ids = set(_members(con, chosen))
        members = [p for p in everyone if p["player_id"] in ids]
        out["playlist"] = {**_playlist_out(row), **_setup_stats(data, members, league_medians, schedule)}
        if row["is_pool"]:
            out["assessment"] = assessment(con, {}, {}, chosen)
            best = {u["out"]["player_id"]: u["targets"][0] for u in out["assessment"]["upgrades"]}
            for p in out["playlist"]["attention"]:
                p["upgrade"] = best.get(p["player_id"])
    return out


# --- NHL data ---

def nhl_status(con, params, body):
    """When each kind of NHL data was last refreshed, and how much of it there is."""
    synced = {r["kind"]: {"at": r["at"], "ok": bool(r["ok"]), "detail": r["detail"]} for r in con.execute("SELECT * FROM nhl_sync")}
    counts = con.execute("""SELECT (SELECT COUNT(*) FROM nhl_standings) AS standings,
                                   (SELECT COUNT(*) FROM nhl_season_stats) AS stats,
                                   (SELECT COUNT(*) FROM nhl_schedule) AS schedule,
                                   (SELECT COUNT(*) FROM nhl_boxscores) AS boxscores""").fetchone()
    return {"kinds": [{"kind": k, **synced.get(k, {"at": None, "ok": None, "detail": None})} for k in nhl.KINDS],
            "counts": dict(counts), "season": nhl.current_season()}


def nhl_sync(con, params, body):
    """Refresh NHL data now: body {"only": [kinds]} for some of it. Takes a while (one request per team)."""
    only = body.get("only")
    if only is not None and (not isinstance(only, list) or set(only) - set(nhl.KINDS)):
        raise ApiError(400, f"only must be a list of: {', '.join(nhl.KINDS)}")
    return {"report": nhl.sync(con, only), **nhl_status(con, {}, {})}


ROUTES = [
    ("GET", r"/api/dashboard", dashboard),
    ("GET", r"/api/teams", teams),
    ("GET", r"/api/players", players),
    ("GET", r"/api/players/(\d+)", player),
    ("GET", r"/api/players/(\d+)/nhl", player_nhl),
    ("POST", r"/api/players/(\d+)/favorite", add_favorite),
    ("DELETE", r"/api/players/(\d+)/favorite", remove_favorite),
    ("GET", r"/api/games", games),
    ("GET", r"/api/games/(\d+)", game),
    ("DELETE", r"/api/games/(\d+)", delete_game),
    ("GET", r"/api/playlists", playlists),
    ("POST", r"/api/playlists", create_playlist),
    ("GET", r"/api/playlists/(\d+)", playlist),
    ("PATCH", r"/api/playlists/(\d+)", update_playlist),
    ("DELETE", r"/api/playlists/(\d+)", delete_playlist),
    ("POST", r"/api/playlists/(\d+)/players", add_to_playlist),
    ("DELETE", r"/api/playlists/(\d+)/players/(\d+)", remove_from_playlist),
    ("GET", r"/api/playlists/(\d+)/assessment", assessment),
    ("GET", r"/api/nhl/status", nhl_status),
    ("POST", r"/api/nhl/sync", nhl_sync),
]
