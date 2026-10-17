"""The web app's SQLite database: every NHL player (from the roster CSVs), the games imported from
export_data.py outputs with each player's stats, favourites and playlists.

    players          one row per NHL player, keyed by the NHL's player_id
    games            one row per imported game (an export folder)
    game_stats       one row per player per game: role, team and the player_stats.csv row as JSON
    favorites        starred players
    playlists        named lists of players; a pool is a playlist with is_pool set and roster slots
    playlist_players the players in each playlist
"""

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path(__file__).resolve().parent / "hockey_vision.db"
ACTIVE_ROSTER = ROOT / "rosters" / "nhl_active_players.csv"

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    team TEXT,
    team_name TEXT,
    number INTEGER,
    position TEXT,
    as_of TEXT
);
CREATE INDEX IF NOT EXISTS players_team ON players(team);
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT UNIQUE NOT NULL,
    label TEXT NOT NULL,
    date TEXT NOT NULL,
    nhl_game_id INTEGER,
    away TEXT,
    home TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0,
    imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS game_stats (
    game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    team TEXT,
    role TEXT NOT NULL,
    stats TEXT NOT NULL,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS game_stats_player ON game_stats(player_id);
CREATE TABLE IF NOT EXISTS favorites (
    player_id INTEGER PRIMARY KEY REFERENCES players(player_id),
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS playlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    is_pool INTEGER NOT NULL DEFAULT 0,
    slots_f INTEGER NOT NULL DEFAULT 12,
    slots_d INTEGER NOT NULL DEFAULT 6,
    slots_g INTEGER NOT NULL DEFAULT 2,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS playlist_players (
    playlist_id INTEGER NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    added_at TEXT NOT NULL,
    PRIMARY KEY (playlist_id, player_id)
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path=DEFAULT_DB):
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def load_roster(con, path=ACTIVE_ROSTER):
    """Add or update every player in a fetch_roster.py CSV. Rows dated earlier than what the
    database already has (e.g. an old game's roster) don't move a player off their current team."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    with con:
        con.executemany(
            """INSERT INTO players (player_id, name, team, team_name, number, position, as_of)
               VALUES (:player_id, :player, :team, :team_name, :number, :position, :as_of)
               ON CONFLICT(player_id) DO UPDATE SET name = excluded.name, team = excluded.team,
                   team_name = excluded.team_name, number = excluded.number, position = excluded.position,
                   as_of = excluded.as_of
               WHERE excluded.as_of >= COALESCE(players.as_of, '')""",
            [{**r, "player_id": int(r["player_id"]), "number": int(r["number"])} for r in rows])
    return rows


def ensure_players(con):
    """Load the active roster into an empty database."""
    if not con.execute("SELECT 1 FROM players LIMIT 1").fetchone() and ACTIVE_ROSTER.exists():
        load_roster(con)


def data_version(con):
    """Changes whenever games are imported or removed, for caching scores."""
    return tuple(con.execute("SELECT COUNT(*), MAX(id), MAX(imported_at) FROM games").fetchone())


def all_game_stats(con):
    rows = con.execute("""SELECT s.game_id, s.player_id, s.team, s.role, s.stats, g.date, g.label, g.away, g.home,
                                 g.is_demo
                          FROM game_stats s JOIN games g ON g.id = s.game_id
                          ORDER BY g.date DESC, g.id DESC""").fetchall()
    return [{**dict(r), "stats": json.loads(r["stats"])} for r in rows]
