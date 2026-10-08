"""Play events from the rink positions: possessions, shots, passes, turnovers and defensive plays.

Everything here works in the 2D rink plane (feet, from homography.py via export_data.py), so the
camera angle doesn't matter. Frames without a usable rink fit are simply gaps. Only the standard
library is used, like export_data.py, which calls analyze_clip() for each clip.

How each event is found:
    possession  The puck is within POSSESSION_FT of a player's skates and isn't flying past
                faster than CARRY_MAX_SPEED_FT_S. A possession is a run of such touches by the same
                player (at least MIN_TOUCHES, gaps up to POSSESSION_GAP_S, no camera cut).
    shot        A possession ends and within SHOT_WINDOW_S the puck is seen moving at least
                SHOT_MIN_SPEED_FT_S on a line that crosses the goal line between the posts (plus
                SHOT_NET_MARGIN_FT), from within SHOT_MAX_DISTANCE_FT of that net. Or the other
                team's goalie gets the puck within SAVE_WINDOW_S of a release near their net.
    pass        Possession passes from one player to a teammate within MAX_TRANSITION_S, the puck
                moving at least MIN_PASS_FT. A pass is
                "good" when it advances the puck, beats defenders, relieves pressure, finds an open
                teammate or leads to a shot (the reasons are listed on the event).
    turnover    Possession passes to the other team. A "takeaway" when the new carrier was within
                STEAL_FT of the old one (the puck was stolen), an "interception" otherwise, and a
                "goalie_recovery" when the other team's goalie picks it up. Rebounds after a shot
                aren't turnovers.
    defensive_play  A carrier under pressure (an opponent within PRESSURE_FT) who has to backtrack
                (loses BACKTRACK_FT along the attacking direction), is slowed (speed drops to
                SLOWED_RATIO of at least SLOWED_FROM_FT_S), or passes the puck away under tight
                pressure or backwards. Credited to the nearest opponent. A possession that ends in a
                takeaway is already a turnover event, credited to the defender who stole the puck.

Which way each team attacks comes from where its goalie (and the other team's) stands on side-view
frames: a team attacks away from its own goalie. When no goalie has been seen, events that need
the direction (backtracks, forward progress, zones) are left out or left blank.
"""

import math
from collections import Counter

# NHL markings, ft (as in rink.py, repeated so this stays standard-library only).
GOAL_LINE_X = 89.0
GOAL_POST_Y = 3.0
BLUE_LINE_X = 25.0

POSSESSION_FT = 6.0          # puck this close to a player's skates is on their stick
CONTESTED_FT = 1.5           # an opponent within this much of the carrier's distance contests the puck
CARRY_MAX_SPEED_FT_S = 45.0  # a puck moving faster than this is a pass or shot going by, not a touch
# Puck velocity is measured between observations VELOCITY_BASELINE_S to MAX_VELOCITY_GAP_S apart: over a
# single frame, a foot of projection jitter alone reads as 60 ft/s.
VELOCITY_BASELINE_S = 0.1
MAX_VELOCITY_GAP_S = 0.3
MAX_PUCK_SPEED_FT_S = 170.0  # the hardest NHL shots are ~150 ft/s; faster is a false puck detection
POSSESSION_GAP_S = 1.0       # touches by the same player this close together are one possession
MIN_TOUCHES = 3              # fewer touches than this is a deflection or a puck passing by
MERGE_GAP_S = 2.0            # two possessions by the same player this close together are one
MAX_TRANSITION_S = 3.0       # possession changes slower than this aren't linked as passes / turnovers
MIN_PASS_FT = 10.0           # the puck moving less than this between teammates is a handoff, not a pass
STEAL_FT = 8.0               # new carrier this close to the old one: the puck was taken off them
SHOT_WINDOW_S = 1.0          # look this long after a release for the puck heading at the net
SHOT_MIN_SPEED_FT_S = 45.0   # slower than this is a pass or a dump, not a shot
SHOT_MAX_DISTANCE_FT = 65.0  # releases further from the net than this are dump-ins
SHOT_NET_MARGIN_FT = 3.0     # a puck line this far outside the posts still counts as at the net
SAVE_WINDOW_S = 1.5          # the goalie getting the puck this soon after a release is a save
BLOCK_WINDOW_S = 0.5         # an opponent skater getting a shot this soon after release blocked it
PRESSURE_FT = 10.0           # an opponent this close to the carrier is pressuring them
TIGHT_PRESSURE_FT = 6.0      # a pass made with an opponent this close was forced
BACKTRACK_FT = 8.0           # carrier giving up this much ground under pressure has backtracked
SLOWED_FROM_FT_S = 12.0      # carrier skating at least this fast ...
SLOWED_RATIO = 0.5           # ... and dropping to this fraction of it under pressure was slowed
SLOWED_WINDOW_S = 1.5
GOOD_PASS_FORWARD_FT = 15.0  # a pass that moves the puck this far up the ice advanced it
OPEN_FT = 15.0               # a receiver with no opponent this close is open
RELIEF_FT = 5.0              # the receiver has this much more space than the pressured passer
SMOOTH_WINDOW = 5            # frames in the moving average for carrier speed
MIN_GOALIE_X = 60.0          # goalies further out than this aren't at their net (pulled, or misprojected)
DIRECTION_WINDOW_S = 900.0   # goalie sightings this close in time say which way a team attacks

# Which of a defensive play's outcomes to report when one possession shows several.
DEFENSE_PRIORITY = ["forced_pass", "forced_backtrack", "slowed"]


def clock(t):
    """Seconds -> 'm:ss.s' video time."""
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _sign(v):
    return 1 if v > 0 else -1 if v < 0 else 0


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def identify_players(clip, tracks):
    """Who each track is. Returns ({(track_id, class): player_id}, {player_id: info}).

    A track with a jersey number is that player (team + number), so every track of theirs, across
    clips too, counts towards the same player. A track without one stays its own unidentified
    player. Referees are left out."""
    ids, info = {}, {}
    for t in tracks:
        if t["role"] == "referee":
            continue
        team_label = t["team_abbrev"] or f"team_{t['team'].lower()}"
        if t["jersey_number"]:
            pid = f"{team_label} #{t['jersey_number']}"
        else:
            pid = f"{team_label} unidentified ({clip} track {t['track_id']})"
        ids[(t["track_id"], t["class"])] = pid
        info.setdefault(pid, {
            "player_id": pid, "team": t["team"], "team_abbrev": t["team_abbrev"],
            "jersey_number": t["jersey_number"], "player_name": t["player_name"], "role": t["role"],
            "identified": bool(t["jersey_number"]),
        })
        if info[pid]["player_name"] is None and t["player_name"]:
            info[pid]["player_name"] = t["player_name"]
    return ids, info


def label(info):
    """'SJS #63 Zack Ostapchuk', for event descriptions."""
    name = f" {info['player_name']}" if info.get("player_name") else ""
    goalie = " (G)" if info["role"] == "goalie" else ""
    return f"{info['player_id']}{name}{goalie}"


def _build_frames(frame_rows, player_rows, puck_rows, ids):
    """Per frame, in order: time, camera segment, view, {player_id: (x, y)} and the puck (x, y)."""
    frames = {}
    seg = 0
    for r in sorted(frame_rows, key=lambda r: r["frame"]):
        seg += bool(r.get("camera_cut"))
        frames[r["frame"]] = {"frame": r["frame"], "t": r["time_s"], "seg": seg, "view": r.get("view"),
                              "players": {}, "puck": None}
    for r in player_rows:
        pid = ids.get((r["track_id"], r["class"]))
        if pid is not None and r["x_ft"] is not None and r["frame"] in frames:
            frames[r["frame"]]["players"].setdefault(pid, (r["x_ft"], r["y_ft"]))
    for r in puck_rows:
        if r["x_ft"] is not None and r["frame"] in frames:
            frames[r["frame"]]["puck"] = (r["x_ft"], r["y_ft"])
    return [frames[k] for k in sorted(frames)]


class AttackDirections:
    """Which way (+1 towards x = +100, -1 towards x = -100) each team attacks at a given time."""

    def __init__(self, frames, info):
        self.obs = {"A": [], "B": []}
        for f in frames:
            if f["view"] not in (None, "side"):  # end-view cameras are always mapped to the right end
                continue
            for pid, (x, _) in f["players"].items():
                p = info[pid]
                if p["role"] != "goalie" or abs(x) < MIN_GOALIE_X:
                    continue
                other = "B" if p["team"] == "A" else "A"
                self.obs[p["team"]].append((f["t"], -_sign(x)))  # attacks away from its own goalie
                self.obs[other].append((f["t"], _sign(x)))

    def attack(self, team, t):
        votes = [s for ot, s in self.obs.get(team, []) if abs(ot - t) <= DIRECTION_WINDOW_S]
        if not votes:
            return None
        total = sum(votes)
        return _sign(total) if abs(total) >= 0.6 * len(votes) else None


def _puck_velocities(frames):
    """{frame index: (vx, vy)} from the latest earlier puck observation at least VELOCITY_BASELINE_S
    back (and no more than MAX_VELOCITY_GAP_S), in the same camera shot."""
    vel = {}
    obs = [i for i, f in enumerate(frames) if f["puck"]]
    for k, b in enumerate(obs):
        fb = frames[b]
        for a in reversed(obs[:k]):
            fa = frames[a]
            dt = fb["t"] - fa["t"]
            if dt > MAX_VELOCITY_GAP_S or fa["seg"] != fb["seg"]:
                break
            if dt >= VELOCITY_BASELINE_S:
                vel[b] = ((fb["puck"][0] - fa["puck"][0]) / dt, (fb["puck"][1] - fa["puck"][1]) / dt)
                break
    return vel


def _possessions(frames, info):
    vel = _puck_velocities(frames)
    touches = []
    for i, f in enumerate(frames):
        if not f["puck"] or not f["players"]:
            continue
        v = vel.get(i)
        if v is not None and math.hypot(*v) > CARRY_MAX_SPEED_FT_S:
            continue
        near = sorted((_dist(f["puck"], xy), pid) for pid, xy in f["players"].items())
        d, pid = near[0]
        if d > POSSESSION_FT:
            continue
        contested = any(od - d <= CONTESTED_FT and info[op]["team"] != info[pid]["team"] for od, op in near[1:])
        touches.append({"i": i, "pid": pid, "contested": contested})

    runs = []
    for tch in touches:
        f = frames[tch["i"]]
        cur = runs[-1] if runs else None
        if (cur and cur["pid"] == tch["pid"] and frames[cur["end_i"]]["seg"] == f["seg"]
                and f["t"] - frames[cur["end_i"]]["t"] <= POSSESSION_GAP_S):
            cur["end_i"] = tch["i"]
            cur["touches"] += 1
            cur["contested"] += tch["contested"]
        else:
            runs.append({"pid": tch["pid"], "start_i": tch["i"], "end_i": tch["i"], "touches": 1,
                         "contested": int(tch["contested"])})
    runs = [r for r in runs if r["touches"] >= MIN_TOUCHES]

    merged = []
    for r in runs:
        prev = merged[-1] if merged else None
        if (prev and prev["pid"] == r["pid"] and frames[prev["end_i"]]["seg"] == frames[r["start_i"]]["seg"]
                and frames[r["start_i"]]["t"] - frames[prev["end_i"]]["t"] <= MERGE_GAP_S):
            prev["end_i"] = r["end_i"]
            prev["touches"] += r["touches"]
            prev["contested"] += r["contested"]
        else:
            merged.append(r)
    return merged


def _nearest_opponent(frame, pid, info, skaters_only=True):
    """(distance, player_id) of the closest opponent to `pid` in this frame, or (None, None)."""
    me = frame["players"].get(pid)
    if me is None:
        return None, None
    team = info[pid]["team"]
    best = (None, None)
    for op, xy in frame["players"].items():
        o = info[op]
        if o["team"] == team or (skaters_only and o["role"] != "skater"):
            continue
        d = _dist(me, xy)
        if best[0] is None or d < best[0]:
            best = (d, op)
    return best


def _smooth(points):
    half = SMOOTH_WINDOW // 2
    out = []
    for i in range(len(points)):
        win = points[max(0, i - half):i + half + 1]
        out.append((sum(p[0] for p in win) / len(win), sum(p[1] for p in win) / len(win)))
    return out


def _zone(x, attack):
    """'offensive', 'neutral' or 'defensive' for a team attacking towards `attack`, or None."""
    if attack is None:
        return None
    a = x * attack
    return "offensive" if a > BLUE_LINE_X else "defensive" if a < -BLUE_LINE_X else "neutral"


class _Clip:
    def __init__(self, clip, frames, info):
        self.clip, self.frames, self.info = clip, frames, info
        self.directions = AttackDirections(frames, info)
        self.events = []

    def t(self, i):
        return self.frames[i]["t"]

    def event(self, type_, subtype, i, pid, description, **fields):
        f = self.frames[i]
        p = self.info[pid]
        xy = f["players"].get(pid) or f["puck"] or (None, None)
        attack = self.directions.attack(p["team"], f["t"])
        e = {"clip": self.clip, "type": type_, "subtype": subtype, "frame": f["frame"],
             "time_s": round(f["t"], 3), "time": clock(f["t"]), "team": p["team"],
             "team_abbrev": p["team_abbrev"], "player_id": pid, "player_name": p["player_name"],
             "x_ft": None if xy[0] is None else round(xy[0], 1), "y_ft": None if xy[1] is None else round(xy[1], 1),
             "zone": None if xy[0] is None else _zone(xy[0], attack)}
        e.update(fields)
        e["description"] = description
        self.events.append(e)
        return e

    # ---- shots

    def shot(self, p, nxt):
        """The shot that ended possession p, or None."""
        f_end = self.frames[p["end_i"]]
        release = f_end["puck"]
        team = self.info[p["pid"]]["team"]
        attack = self.directions.attack(team, f_end["t"])
        nxt_same = nxt is not None and self.frames[nxt["start_i"]]["seg"] == f_end["seg"]
        limit = f_end["t"] + SHOT_WINDOW_S
        if nxt_same:
            limit = min(limit, self.t(nxt["start_i"]))

        best = None  # (speed, goal_x, y at the goal line)
        if abs(release[0]) < GOAL_LINE_X:
            j = p["end_i"] + 1
            while j < len(self.frames) and self.frames[j]["t"] <= limit and self.frames[j]["seg"] == f_end["seg"]:
                puck = self.frames[j]["puck"]
                dt = self.frames[j]["t"] - f_end["t"]
                if puck and dt >= VELOCITY_BASELINE_S:
                    vx, vy = (puck[0] - release[0]) / dt, (puck[1] - release[1]) / dt
                    speed = math.hypot(vx, vy)
                    if SHOT_MIN_SPEED_FT_S <= speed <= MAX_PUCK_SPEED_FT_S and vx != 0:
                        goal_x = GOAL_LINE_X * _sign(vx)
                        y_cross = release[1] + vy * (goal_x - release[0]) / vx
                        if (abs(y_cross) <= GOAL_POST_Y + SHOT_NET_MARGIN_FT
                                and _dist(release, (goal_x, 0)) <= SHOT_MAX_DISTANCE_FT
                                and (best is None or speed > best[0])):
                            best = (speed, goal_x, y_cross)
                j += 1

        # The other team's goalie getting the puck right after a release near their net.
        goalie_save = False
        if nxt_same and self.t(nxt["start_i"]) - f_end["t"] <= SAVE_WINDOW_S:
            g = self.info[nxt["pid"]]
            gxy = self.frames[nxt["start_i"]]["players"].get(nxt["pid"])
            if g["role"] == "goalie" and g["team"] != team and gxy is not None and abs(gxy[0]) >= MIN_GOALIE_X:
                goal = (GOAL_LINE_X * _sign(gxy[0]), 0.0)
                if _dist(release, goal) <= SHOT_MAX_DISTANCE_FT:
                    goalie_save = True
                    if best is None:
                        best = (None, goal[0], None)
        if best is None:
            return None
        speed, goal_x, y_cross = best
        if attack is not None and _sign(goal_x) != attack and f_end["view"] == "side":
            return None  # towards their own net: a clearance or a pass back to the goalie

        outcome, by = "unknown", None
        if self._puck_in_net(p["end_i"], goal_x, limit + SAVE_WINDOW_S):
            outcome = "possible_goal"
        elif goalie_save:
            outcome, by = "saved", nxt["pid"]
        elif nxt_same and self.t(nxt["start_i"]) - f_end["t"] <= SAVE_WINDOW_S:
            by = nxt["pid"]
            n = self.info[by]
            if n["team"] == team:
                outcome = "rebound_recovered"
            elif n["role"] == "goalie":
                outcome = "saved"
            elif self.t(nxt["start_i"]) - f_end["t"] <= BLOCK_WINDOW_S:
                outcome = "blocked"
            else:
                outcome = "recovered_by_opponent"

        distance = _dist(release, (goal_x, 0.0))
        on_target = y_cross is not None and abs(y_cross) <= GOAL_POST_Y
        shooter = self.info[p["pid"]]
        text = (f"{label(shooter)} shot at the {'right' if goal_x > 0 else 'left'} net from "
                f"{distance:.0f} ft at {clock(f_end['t'])}")
        if speed is not None:
            text += f", puck at {speed:.0f} ft/s"
        text += {"saved": f"; saved by {label(self.info[by]) if by else 'the goalie'}",
                 "possible_goal": "; the puck was seen in the goal mouth (possible goal)",
                 "blocked": f"; blocked by {label(self.info[by]) if by else 'a defender'}",
                 "rebound_recovered": f"; rebound recovered by teammate {label(self.info[by]) if by else ''}",
                 "recovered_by_opponent": f"; recovered by {label(self.info[by]) if by else 'the other team'}",
                 "unknown": ""}[outcome]
        return self.event("shot", outcome, p["end_i"], p["pid"], text,
                          distance_ft=round(distance, 1), puck_speed_ft_s=None if speed is None else round(speed, 1),
                          target_net="right" if goal_x > 0 else "left",
                          on_target=on_target if y_cross is not None else None,
                          other_player_id=by, other_player_name=self.info[by]["player_name"] if by else None)

    def _puck_in_net(self, start_i, goal_x, until):
        seg = self.frames[start_i]["seg"]
        for f in self.frames[start_i + 1:]:
            if f["t"] > until or f["seg"] != seg:
                break
            if f["puck"] and _sign(f["puck"][0]) == _sign(goal_x) and abs(f["puck"][0]) >= GOAL_LINE_X \
                    and abs(f["puck"][1]) <= GOAL_POST_Y:
                return True
        return False

    # ---- passes and turnovers

    def pass_(self, p, nxt, shot_next):
        fa, fb = self.frames[p["end_i"]], self.frames[nxt["start_i"]]
        passer, receiver = self.info[p["pid"]], self.info[nxt["pid"]]
        attack = self.directions.attack(passer["team"], fa["t"])
        start, end = fa["puck"], fb["puck"]
        forward = None if attack is None else (end[0] - start[0]) * attack
        pressure, presser = _nearest_opponent(fa, p["pid"], self.info)
        space, _ = _nearest_opponent(fb, nxt["pid"], self.info)

        bypassed = 0
        if attack is not None and forward is not None and forward > 0:
            lo, hi = start[0] * attack, end[0] * attack
            bypassed = sum(1 for op, xy in fa["players"].items()
                           if self.info[op]["team"] != passer["team"] and self.info[op]["role"] == "skater"
                           and lo < xy[0] * attack < hi)

        reasons = []
        if forward is not None and forward >= GOOD_PASS_FORWARD_FT:
            reasons.append("advanced_puck")
        if bypassed:
            reasons.append("beat_defenders")
        if pressure is not None and space is not None and pressure <= PRESSURE_FT and space >= pressure + RELIEF_FT:
            reasons.append("relieved_pressure")
        if space is not None and space >= OPEN_FT:
            reasons.append("found_open_teammate")
        if shot_next:
            reasons.append("led_to_shot")

        distance = _dist(start, end)
        text = f"{label(passer)} passed to {label(receiver)} at {clock(fa['t'])} ({distance:.0f} ft"
        if forward is not None:
            text += f", {abs(forward):.0f} ft {'up' if forward >= 0 else 'back down'} the ice"
        text += ")"
        if reasons:
            text += "; good pass: " + ", ".join(r.replace("_", " ") for r in reasons)
        return self.event("pass", "good" if reasons else "completed", p["end_i"], p["pid"], text,
                          other_player_id=nxt["pid"], other_player_name=receiver["player_name"],
                          end_time_s=round(fb["t"], 3), distance_ft=round(distance, 1),
                          forward_ft=None if forward is None else round(forward, 1),
                          passer_pressure_ft=None if pressure is None else round(pressure, 1),
                          pressured_by=presser if pressure is not None and pressure <= PRESSURE_FT else None,
                          receiver_space_ft=None if space is None else round(space, 1),
                          opponents_bypassed=bypassed, good=bool(reasons), good_reasons=reasons)

    def turnover(self, p, nxt):
        fa, fb = self.frames[p["end_i"]], self.frames[nxt["start_i"]]
        loser, gainer = self.info[p["pid"]], self.info[nxt["pid"]]
        gaps = [_dist(f["players"][p["pid"]], f["players"][nxt["pid"]]) for f in (fa, fb)
                if p["pid"] in f["players"] and nxt["pid"] in f["players"]]
        gap = min(gaps) if gaps else None
        if gainer["role"] == "goalie":
            subtype = "goalie_recovery"
            text = f"{label(gainer)} picked up the puck from {label(loser)} at {clock(fb['t'])}"
        elif gap is not None and gap <= STEAL_FT:
            subtype = "takeaway"
            text = f"{label(gainer)} stole the puck from {label(loser)} at {clock(fb['t'])}"
        else:
            subtype = "interception"
            text = f"{label(gainer)} intercepted the puck from {label(loser)} at {clock(fb['t'])}"
        return self.event("turnover", subtype, nxt["start_i"], nxt["pid"], text,
                          other_player_id=p["pid"], other_player_name=loser["player_name"],
                          distance_between_ft=None if gap is None else round(gap, 1))

    # ---- defensive plays

    def defense(self, p, ended_by):
        """The strongest defensive play against possession p, or None. `ended_by` is the pass or
        turnover event that ended it, if any."""
        pid, team = p["pid"], self.info[p["pid"]]["team"]
        carrier = self.info[pid]
        found = {}

        if ended_by and ended_by["type"] == "turnover" and ended_by["subtype"] == "takeaway":
            return None  # reported as the takeaway
        if ended_by and ended_by["type"] == "pass" and ended_by["pressured_by"]:
            pressure, forward = ended_by["passer_pressure_ft"], ended_by["forward_ft"]
            if pressure <= TIGHT_PRESSURE_FT or (forward is not None and forward <= 0):
                found["forced_pass"] = (ended_by["pressured_by"], p["end_i"], pressure)

        idx = [i for i in range(p["start_i"], p["end_i"] + 1) if pid in self.frames[i]["players"]]
        if len(idx) >= 3:
            path = _smooth([self.frames[i]["players"][pid] for i in idx])
            opp = [_nearest_opponent(self.frames[i], pid, self.info) for i in idx]
            attack = self.directions.attack(team, self.t(p["start_i"]))
            if attack is not None:
                best_k = 0
                for k in range(1, len(idx)):
                    if path[k][0] * attack > path[best_k][0] * attack:
                        best_k = k
                    elif (path[best_k][0] - path[k][0]) * attack >= BACKTRACK_FT:
                        close = [(d, op, idx[m]) for m, (d, op) in enumerate(opp[best_k:k + 1], best_k)
                                 if d is not None and d <= PRESSURE_FT]
                        if close:
                            d, op, i = min(close)
                            found["forced_backtrack"] = (op, i, d)
                        break
            speeds = []
            for k in range(1, len(idx)):
                dt = self.t(idx[k]) - self.t(idx[k - 1])
                speeds.append(_dist(path[k], path[k - 1]) / dt if dt > 0 else None)
            for a, sa in enumerate(speeds):
                if sa is None or sa < SLOWED_FROM_FT_S or "slowed" in found:
                    continue
                for b in range(a + 1, len(speeds)):
                    if self.t(idx[b + 1]) - self.t(idx[a + 1]) > SLOWED_WINDOW_S:
                        break
                    d, op = opp[b + 1]
                    if speeds[b] is not None and speeds[b] <= SLOWED_RATIO * sa and d is not None and d <= PRESSURE_FT:
                        found["slowed"] = (op, idx[b + 1], d)
                        break

        for kind in DEFENSE_PRIORITY:
            if kind not in found:
                continue
            defender, i, d = found[kind]
            what = {"forced_pass": "forced a pass from",
                    "forced_backtrack": "forced a backtrack from", "slowed": "slowed down"}[kind]
            text = f"{label(self.info[defender])} {what} {label(carrier)} at {clock(self.t(i))}"
            if d is not None:
                text += f" (within {d:.0f} ft)"
            return self.event("defensive_play", kind, i, defender, text,
                              other_player_id=pid, other_player_name=carrier["player_name"],
                              gap_ft=None if d is None else round(d, 1))
        return None


def analyze_clip(clip, frame_rows, player_rows, puck_rows, tracks):
    """Possessions and play events for one clip. Returns (possession rows, events, player info)."""
    ids, info = identify_players(clip, tracks)
    frames = _build_frames(frame_rows, player_rows, puck_rows, ids)
    c = _Clip(clip, frames, info)
    poss = _possessions(frames, info)

    shots = [c.shot(p, poss[k + 1] if k + 1 < len(poss) else None) for k, p in enumerate(poss)]
    ended_by = [None] * len(poss)
    for k, (p, nxt) in enumerate(zip(poss, poss[1:])):
        if frames[p["end_i"]]["seg"] != frames[nxt["start_i"]]["seg"] or \
                frames[nxt["start_i"]]["t"] - frames[p["end_i"]]["t"] > MAX_TRANSITION_S:
            continue
        if shots[k] is not None:
            continue  # what happens after a shot is its outcome (save, rebound, block)
        if info[p["pid"]]["team"] == info[nxt["pid"]]["team"]:
            if _dist(frames[p["end_i"]]["puck"], frames[nxt["start_i"]]["puck"]) >= MIN_PASS_FT:
                ended_by[k] = c.pass_(p, nxt, shots[k + 1] is not None)
        else:
            ended_by[k] = c.turnover(p, nxt)
    for p, e in zip(poss, ended_by):
        c.defense(p, e)

    rows = []
    for k, p in enumerate(poss):
        a, b = frames[p["start_i"]], frames[p["end_i"]]
        pl = info[p["pid"]]
        end = "shot" if shots[k] else ended_by[k]["type"] if ended_by[k] else "lost_track"
        if ended_by[k] is not None and ended_by[k]["type"] == "turnover":
            end = f"turnover_{ended_by[k]['subtype']}"
        rows.append({
            "clip": clip, "player_id": p["pid"], "team": pl["team"], "team_abbrev": pl["team_abbrev"],
            "jersey_number": pl["jersey_number"], "player_name": pl["player_name"],
            "start_frame": a["frame"], "end_frame": b["frame"], "start_time_s": a["t"], "end_time_s": b["t"],
            "duration_s": round(b["t"] - a["t"], 3), "touches": p["touches"], "contested_touches": p["contested"],
            "start_x_ft": a["puck"][0], "start_y_ft": a["puck"][1], "end_x_ft": b["puck"][0], "end_y_ft": b["puck"][1],
            "zone": _zone(a["puck"][0], c.directions.attack(pl["team"], a["t"])),
            "ended_by": end,
        })
    events = sorted(c.events, key=lambda e: (e["time_s"], e["type"]))
    directions = {team: c.directions.attack(team, frames[len(frames) // 2]["t"]) if frames else None
                  for team in ("A", "B")}
    return rows, events, info, directions


def player_stats(info, events, possessions, presence):
    """One row per player. `presence` is {player_id: {"frames": n, "frames_on_rink": n,
    "seconds": s, "seconds_on_rink": s, "distance_ft": d}}."""
    stats = {pid: Counter() for pid in info}
    for e in events:
        pid, other = e["player_id"], e.get("other_player_id")
        s = stats[pid]
        if e["type"] == "shot":
            s["shots"] += 1
            s["shots_on_net"] += e["subtype"] in ("saved", "possible_goal") or bool(e.get("on_target"))
            s["possible_goals"] += e["subtype"] == "possible_goal"
            s["shots_blocked_by_opponent"] += e["subtype"] == "blocked"
            if other and e["subtype"] == "saved":
                stats[other]["saves"] += 1
            if other and e["subtype"] == "blocked":
                stats[other]["blocked_shots"] += 1
        elif e["type"] == "pass":
            s["passes_completed"] += 1
            s["good_passes"] += e["good"]
            s["pass_distance_ft"] += e["distance_ft"]
            stats[other]["passes_received"] += 1
        elif e["type"] == "turnover":
            s[{"takeaway": "takeaways", "interception": "interceptions",
               "goalie_recovery": "goalie_recoveries"}[e["subtype"]]] += 1
            stats[other]["turnovers"] += 1
            stats[other][f"lost_puck_{e['subtype']}"] += 1
        elif e["type"] == "defensive_play":
            s["defensive_plays"] += 1
            s[f"defense_{e['subtype']}"] += 1
            stats[other]["times_defended"] += 1
    for p in possessions:
        s = stats[p["player_id"]]
        s["possessions"] += 1
        s["possession_time_s"] += p["duration_s"]
        s["puck_touches"] += p["touches"]

    rows = []
    for pid, p in info.items():
        s, pr = stats[pid], presence.get(pid, {})
        rows.append({
            **p,
            "time_detected_s": round(pr.get("seconds", 0.0), 2),
            "time_on_rink_s": round(pr.get("seconds_on_rink", 0.0), 2),
            "frames_detected": pr.get("frames", 0),
            "distance_ft": round(pr.get("distance_ft", 0.0), 1),
            "possessions": s["possessions"], "possession_time_s": round(s["possession_time_s"], 2),
            "puck_touches": s["puck_touches"],
            "shots": s["shots"], "shots_on_net": s["shots_on_net"], "possible_goals": s["possible_goals"],
            "shots_blocked_by_opponent": s["shots_blocked_by_opponent"],
            "passes_completed": s["passes_completed"], "good_passes": s["good_passes"],
            "passes_received": s["passes_received"],
            "mean_pass_distance_ft": round(s["pass_distance_ft"] / s["passes_completed"], 1) if s["passes_completed"] else None,
            "takeaways": s["takeaways"], "interceptions": s["interceptions"],
            "turnovers": s["turnovers"], "defensive_plays": s["defensive_plays"],
            "defense_forced_pass": s["defense_forced_pass"],
            "defense_forced_backtrack": s["defense_forced_backtrack"], "defense_slowed": s["defense_slowed"],
            "times_defended": s["times_defended"], "blocked_shots": s["blocked_shots"], "saves": s["saves"],
            "goalie_recoveries": s["goalie_recoveries"],
        })
    rows.sort(key=lambda r: (not r["identified"], r["team"], r["team_abbrev"] or "",
                             int(r["jersey_number"]) if r["jersey_number"] else 0, r["player_id"]))
    return rows
