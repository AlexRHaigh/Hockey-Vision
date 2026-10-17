"""The web app's JSON API, as plain functions of a database connection. server.py routes HTTP
requests to them (ROUTES); tests call them directly.

Scores depend on every game in the database (scoring.calibrate), so they are computed together and
cached until games are imported or removed.
"""

import threading

from . import db, scoring


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
    rows = db.all_game_stats(con)
    points = [(r["role"], *scoring_inputs(r)) for r in rows]
    calibration = scoring.calibrate(points)
    by_player, by_game = {}, {}
    for r in rows:
        entry = {"game_id": r["game_id"], "date": r["date"], "label": r["label"], "away": r["away"], "home": r["home"],
                 "is_demo": bool(r["is_demo"]), "team": r["team"], "role": r["role"], "player_id": r["player_id"],
                 **scoring.game_score(r["stats"], r["role"], calibration), "stats": r["stats"]}
        by_player.setdefault(r["player_id"], []).append(entry)
        by_game.setdefault(r["game_id"], []).append(entry)
    value = {"by_player": by_player, "by_game": by_game, "calibration": calibration,
             "forms": {pid: scoring.form(games) for pid, games in by_player.items()}}
    with _cache_lock:
        _cache.update(key=key, value=value)
    return value


def scoring_inputs(row):
    return scoring.impact_points(row["stats"], row["role"])[0], scoring.minutes_detected(row["stats"])


def _player(row, data, favorites=frozenset()):
    f = data["forms"].get(row["player_id"]) or scoring.form([])
    return {"player_id": row["player_id"], "name": row["name"], "team": row["team"], "team_name": row["team_name"],
            "number": row["number"], "position": row["position"], "group": scoring.position_group(row["position"]),
            "favorite": row["player_id"] in favorites, "form": f, "recommendation": scoring.recommendation(f)}


def _favorites(con):
    return {r[0] for r in con.execute("SELECT player_id FROM favorites")}


def _get_player_row(con, player_id):
    row = con.execute("SELECT * FROM players WHERE player_id = ?", (player_id,)).fetchone()
    if not row:
        raise ApiError(404, f"No player {player_id}")
    return row


# --- players and teams ---

def teams(con, params, body):
    return [dict(r) for r in con.execute(
        """SELECT team, team_name, COUNT(*) AS players FROM players WHERE team IS NOT NULL
           GROUP BY team, team_name ORDER BY team_name""")]


def players(con, params, body):
    """Players filtered by ?team=, ?q= (name), ?group= (F/D/G), ?favorites=1, ?rated=1; ?sort=form|name|number."""
    sql, args = "SELECT * FROM players WHERE 1", []
    if params.get("team"):
        sql += " AND team = ?"
        args.append(params["team"].upper())
    if params.get("q"):
        sql += " AND name LIKE ?"
        args.append(f"%{params['q']}%")
    if params.get("favorites") == "1":
        sql += " AND player_id IN (SELECT player_id FROM favorites)"
    data, favs = scored(con), _favorites(con)
    out = [_player(r, data, favs) for r in con.execute(sql + " ORDER BY name", args)]
    if params.get("group"):
        out = [p for p in out if p["group"] == params["group"].upper()]
    if params.get("rated") == "1":
        out = [p for p in out if p["form"]["games"]]
    sort = params.get("sort", "name")
    if sort == "form":
        out.sort(key=lambda p: (p["form"]["form"] is None, -(p["form"]["form"] or 0)))
    elif sort == "number":
        out.sort(key=lambda p: (p["group"] != "F", p["group"] == "G", p["number"] or 0))
    limit = int(params.get("limit", 0) or 0)
    return out[:limit] if limit else out


def player(con, params, body, player_id):
    row = _get_player_row(con, player_id)
    data = scored(con)
    p = _player(row, data, _favorites(con))
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
    return {**dict(row), "is_demo": bool(row["is_demo"]),
            "players": [{**{k: v for k, v in e.items() if k != "stats"}, **names.get(e["player_id"], {}),
                         "stats": e["stats"]} for e in entries]}


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
    data, favs = scored(con), _favorites(con)
    ids = _members(con, playlist_id)
    rows = {r["player_id"]: r for r in con.execute(
        f"SELECT * FROM players WHERE player_id IN ({', '.join('?' * len(ids))})", ids)} if ids else {}
    members = [_player(rows[i], data, favs) for i in ids if i in rows]
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
        elif p["form"]["games"]:
            others.append(p)
    result = scoring.assess_pool(pool, others, {"F": row["slots_f"], "D": row["slots_d"], "G": row["slots_g"]})
    return {"playlist_id": playlist_id, **result}


# --- overview ---

def summary(con, params, body):
    data = scored(con)
    counts = con.execute("""SELECT (SELECT COUNT(*) FROM players) AS players,
                                   (SELECT COUNT(DISTINCT team) FROM players) AS teams,
                                   (SELECT COUNT(*) FROM games) AS games,
                                   (SELECT COUNT(*) FROM games WHERE is_demo) AS demo_games,
                                   (SELECT COUNT(*) FROM favorites) AS favorites,
                                   (SELECT COUNT(*) FROM playlists) AS playlists""").fetchone()
    return {**dict(counts), "rated_players": len(data["forms"]), "calibration": data["calibration"],
            "thresholds": {"trade_below": scoring.TRADE_BELOW, "watch_below": scoring.WATCH_BELOW,
                           "core_above": scoring.CORE_ABOVE, "min_games": scoring.MIN_GAMES}}


ROUTES = [
    ("GET", r"/api/summary", summary),
    ("GET", r"/api/teams", teams),
    ("GET", r"/api/players", players),
    ("GET", r"/api/players/(\d+)", player),
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
]
