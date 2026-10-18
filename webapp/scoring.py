"""Hockey-Vision scores: how well a player played in one game (0-100), their form over recent games,
whether a pool should keep or trade them, and how strong a pool is overall.

Everything here is a pure function of player_stats.csv rows (export_data.py), so the rules can be
read and tuned in one place.

Game score
    Each counted stat is worth some impact points (SKATER_WEIGHTS / GOALIE_WEIGHTS: a takeaway +2, a
    turnover -2, ...). Broadcast video only shows each player some of the time, so the points are
    turned into a rate per minute the player was detected, shrunk towards the league's typical rate
    (SHRINK_MINUTES) so a player seen for 20 seconds can't score 100 off one shot. The rate is then
    compared with every other player-game in the database of the same role (median and spread),
    and mapped onto 0-100 with a logistic curve: 50 is a typical game, 73 one spread better, 88 two.

Form
    A recency-weighted mean of the last FORM_GAMES scores, games with more time detected counting
    more. Trend is the last 3 games against the ones before them.

Trade call (per pool player)
    hold    fewer than MIN_GAMES games, too early to judge
    trade   form below TRADE_BELOW
    watch   form below WATCH_BELOW, or a trend drop of TREND_DROP or more
    keep    otherwise ("core" from CORE_ABOVE)

Pool assessment
    Strength (mean form of the pool's rated players, graded A-F), depth per position group against
    the pool's slots, trade candidates and the best available upgrades, all folded into a 0-100
    trade urgency and a verdict: hold, make a targeted trade, or be active.
"""

import math
from statistics import mean, median

# Impact points per counted stat. Shots on net and possible goals are also counted in shots, so a
# shot on net is worth 0.5 + 1.5 and a possible goal 0.5 + 1.5 + 5.
SKATER_WEIGHTS = {
    "possible_goals": ("Possible goals", 5.0),
    "shots_on_net": ("Shots on net", 1.5),
    "shots": ("Shots", 0.5),
    "good_passes": ("Good passes", 1.0),
    "passes_completed": ("Passes completed", 0.5),
    "passes_received": ("Passes received", 0.2),
    "takeaways": ("Takeaways", 2.0),
    "interceptions": ("Interceptions", 1.5),
    "blocked_shots": ("Blocked shots", 1.5),
    "defensive_plays": ("Defensive plays", 1.0),
    "possession_time_s": ("Possession time (s)", 0.05),
    "turnovers": ("Turnovers", -2.0),
    "shots_blocked_by_opponent": ("Shots blocked by opponent", -0.5),
    "times_defended": ("Times defended", -0.3),
}
GOALIE_WEIGHTS = {
    "saves": ("Saves", 1.0),
    "goalie_recoveries": ("Puck recoveries", 0.3),
    "passes_completed": ("Passes completed", 0.2),
    "goals_against": ("Possible goals against", -3.0),
    "turnovers": ("Turnovers", -1.0),
}

# Typical impact points per minute detected, and their spread, until the database holds
# MIN_CALIBRATION_ROWS player-games of a role to measure them from.
DEFAULT_CALIBRATION = {"skater": (1.5, 1.2), "goalie": (0.8, 0.8)}
MIN_CALIBRATION_ROWS = 30
MIN_CALIBRATION_MINUTES = 0.5
MIN_SPREAD = {"skater": 0.3, "goalie": 0.2}
SHRINK_MINUTES = 3.0       # minutes of league-typical play every rate is blended with
FULL_MINUTES = 4.0         # minutes detected for a game to count fully in form and be "high" confidence

FORM_GAMES = 5
FORM_DECAY = 0.8           # each older game counts this much of the one after it
MIN_GAMES = 3
TRADE_BELOW = 40
WATCH_BELOW = 46
CORE_ABOVE = 60
TREND_DROP = -10

UPGRADE_MARGIN = 8         # form points an available player must beat a pool player by
GRADES = [(58, "A"), (54, "B"), (50, "C"), (46, "D"), (-1, "F")]
POSITION_GROUPS = {"Center": "F", "Left wing": "F", "Right wing": "F", "Defense": "D", "Goalie": "G"}
GROUP_NAMES = {"F": "Forwards", "D": "Defense", "G": "Goalies"}
GROUP_UNITS = {"F": ("forward", "forwards"), "D": ("defenseman", "defensemen"), "G": ("goalie", "goalies")}


def position_group(position):
    return POSITION_GROUPS.get(position, "F")


def _weights(role):
    return GOALIE_WEIGHTS if role == "goalie" else SKATER_WEIGHTS


def impact_points(stats, role):
    """(total points, [component dicts]) for one game's stats."""
    components, total = [], 0.0
    for key, (label, weight) in _weights(role).items():
        count = float(stats.get(key) or 0)
        if count:
            components.append({"stat": key, "label": label, "count": round(count, 2), "weight": weight,
                               "points": round(count * weight, 2)})
            total += count * weight
    components.sort(key=lambda c: -abs(c["points"]))
    return total, components


def minutes_detected(stats):
    return float(stats.get("time_detected_s") or 0) / 60


def calibrate(rows):
    """{role: (typical rate, spread)} from (role, points, minutes) of every player-game: the median
    points per minute and its scaled median absolute deviation."""
    out = dict(DEFAULT_CALIBRATION)
    for role in out:
        rates = [p / m for r, p, m in rows if r == role and m >= MIN_CALIBRATION_MINUTES]
        if len(rates) >= MIN_CALIBRATION_ROWS:
            mid = median(rates)
            spread = 1.4826 * median(abs(x - mid) for x in rates)
            out[role] = (mid, max(spread, MIN_SPREAD[role]))
    return out


def confidence(minutes):
    return "high" if minutes >= FULL_MINUTES else "medium" if minutes >= 1 else "low"


def game_score(stats, role, calibration):
    """One player's 0-100 score for one game, with the components that make it up."""
    typical, spread = calibration.get(role, DEFAULT_CALIBRATION["skater"])
    points, components = impact_points(stats, role)
    minutes = minutes_detected(stats)
    rate = (points + typical * SHRINK_MINUTES) / (minutes + SHRINK_MINUTES)
    z = (rate - typical) / spread
    return {"score": round(100 / (1 + math.exp(-z)), 1), "points": round(points, 2),
            "minutes": round(minutes, 2), "rate": round(rate, 3), "confidence": confidence(minutes),
            "components": components}


def form(games):
    """A player's form from their scored games, newest first: {form, season, trend, games, last}."""
    if not games:
        return {"form": None, "season": None, "trend": None, "games": 0, "last": None}
    recent = games[:FORM_GAMES]
    weights = [FORM_DECAY ** i * (0.5 + 0.5 * min(1.0, g["minutes"] / FULL_MINUTES)) for i, g in enumerate(recent)]
    value = sum(w * g["score"] for w, g in zip(weights, recent)) / sum(weights)
    trend = None
    if len(games) >= 4:
        trend = round(mean(g["score"] for g in games[:3]) - mean(g["score"] for g in games[3:3 + FORM_GAMES]), 1)
    return {"form": round(value, 1), "season": round(mean(g["score"] for g in games), 1), "trend": trend,
            "games": len(games), "last": games[0]["score"]}


def recommendation(f):
    """Keep, watch, trade or hold for a player with form summary `f`."""
    n, value, trend = f["games"], f["form"], f["trend"]
    if n == 0:
        return {"action": "hold", "label": "Hold", "reason": "No games analysed yet."}
    if n < MIN_GAMES:
        return {"action": "hold", "label": "Hold",
                "reason": f"Only {n} game{'s' * (n != 1)} analysed (form {value:.0f}); too early to judge."}
    if value < TRADE_BELOW:
        return {"action": "trade", "label": "Trade",
                "reason": f"Form {value:.0f} is well below a typical game (50) over the last {min(n, FORM_GAMES)} games."}
    if value < WATCH_BELOW:
        return {"action": "watch", "label": "Watch",
                "reason": f"Form {value:.0f} is below a typical game (50); shop this player if it doesn't pick up."}
    if trend is not None and trend <= TREND_DROP:
        return {"action": "watch", "label": "Watch",
                "reason": f"Form {value:.0f} is fine, but the last 3 games are {-trend:.0f} points below the ones before."}
    if value >= CORE_ABOVE:
        return {"action": "keep", "label": "Keep (core)", "reason": f"Form {value:.0f}: one of the pool's drivers."}
    return {"action": "keep", "label": "Keep", "reason": f"Form {value:.0f} is at or above a typical game."}


def grade(strength):
    return next(g for floor, g in GRADES if strength >= floor)


def assess_pool(pool, others, slots):
    """Assess a pool.

    `pool` and `others` are player dicts with "player_id", "name", "team", "position", "form" (a
    form() summary), "recommendation" and optionally "on_roster" (0 when on no NHL roster); `others`
    are the players outside the pool that could be traded for. `slots` is
    {"F": n, "D": n, "G": n}, how many of each the pool's league rosters.
    """
    rated = [p for p in pool if p["form"]["games"]]
    strength = round(mean(p["form"]["form"] for p in rated), 1) if rated else None
    league = sorted(p["form"]["form"] for p in pool + others if p["form"]["games"])
    league_median = median(league) if league else None
    above = sum(p["form"]["form"] >= league_median for p in rated) if league else 0

    depth = []
    for g in ("F", "D", "G"):
        have = sum(position_group(p["position"]) == g for p in pool)
        depth.append({"group": g, "name": GROUP_NAMES[g], "have": have, "need": slots.get(g, 0),
                      "gap": max(0, slots.get(g, 0) - have)})

    by_action = {a: [p for p in pool if p["recommendation"]["action"] == a] for a in ("keep", "watch", "trade", "hold")}
    # Weakest pool players first; each available player is suggested once, so the list spreads
    # over different targets rather than naming the league's best three for everyone.
    upgrades, suggested = [], set()
    for p in sorted(by_action["trade"] + by_action["watch"], key=lambda p: p["form"]["form"]):
        group = position_group(p["position"])
        targets = sorted((o for o in others if position_group(o["position"]) == group
                          and o["player_id"] not in suggested and o["form"]["games"] >= MIN_GAMES
                          and o["form"]["form"] >= p["form"]["form"] + UPGRADE_MARGIN),
                         key=lambda o: -o["form"]["form"])[:3]
        suggested.update(o["player_id"] for o in targets)
        if targets:
            upgrades.append({"out": _brief(p), "targets": [{**_brief(o), "gain": round(o["form"]["form"] - p["form"]["form"], 1)}
                                                           for o in targets]})

    reasons, urgency = [], 0.0
    if not pool:
        return {"strength": None, "grade": None, "rated": 0, "size": 0, "depth": depth, "counts": {},
                "upgrades": [], "urgency": 0, "verdict": "empty", "verdict_label": "Add players to assess this pool",
                "reasons": ["The pool has no players yet."], "league_median": league_median, "above_median": 0}
    if rated:
        n = len(rated)
        trade_share, watch_share = len(by_action["trade"]) / n, len(by_action["watch"]) / n
        urgency += 70 * trade_share + 25 * watch_share
        urgency += max(0.0, 52 - strength) * 3
        if by_action["trade"]:
            reasons.append(f"{len(by_action['trade'])} player{'s' * (len(by_action['trade']) != 1)} "
                           f"rated Trade: {', '.join(p['name'] for p in by_action['trade'][:4])}.")
        if by_action["watch"]:
            reasons.append(f"{len(by_action['watch'])} on Watch.")
        reasons.append(f"Pool strength {strength:.0f} (grade {grade(strength)}); {above} of {n} rated players "
                       f"are above the league's median form ({league_median:.0f}).")
    if upgrades:
        best = max(t["gain"] for u in upgrades for t in u["targets"])
        urgency += min(20.0, best)
        reasons.append(f"Upgrades are available: up to +{best:.0f} form at the same position.")
    gaps = [d for d in depth if d["gap"]]
    if gaps:
        urgency += 6 * sum(d["gap"] for d in gaps)
        reasons.append("Roster gaps: " + ", ".join(f"{d['gap']} {GROUP_UNITS[d['group']][d['gap'] != 1]}" for d in gaps)
                       + " short.")
    off_roster = [p for p in pool if p.get("on_roster") == 0]
    if off_roster:
        urgency += 8 * len(off_roster)
        reasons.append(f"Not on an NHL roster (sent down, waived or released): {', '.join(p['name'] for p in off_roster[:4])}.")
    unrated = len(pool) - len(rated)
    if unrated:
        reasons.append(f"{unrated} player{'s' * (unrated != 1)} with no analysed games yet.")
    urgency = round(min(100.0, urgency))

    if not rated:
        verdict, label = "hold", "Hold: no games analysed for this pool yet"
    elif urgency >= 55:
        verdict, label = "active", "Be active: trade to improve this pool"
    elif urgency >= 30 or (by_action["trade"] and upgrades):
        verdict, label = "targeted", "Make a targeted trade"
    else:
        verdict, label = "hold", "Hold: the pool is in good shape"
    return {"strength": strength, "grade": grade(strength) if strength is not None else None, "rated": len(rated),
            "size": len(pool), "league_median": league_median, "above_median": above, "depth": depth,
            "counts": {a: len(v) for a, v in by_action.items()}, "upgrades": upgrades, "urgency": urgency,
            "verdict": verdict, "verdict_label": label, "reasons": reasons}


def _brief(p):
    return {"player_id": p["player_id"], "name": p["name"], "team": p["team"], "position": p["position"],
            "form": p["form"]["form"], "games": p["form"]["games"]}
