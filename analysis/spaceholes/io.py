"""Loading and caching of SkillCorner tracking + events, oriented into the attack frame."""

import gzip
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CACHE_DIR, MATCH_DIR

EVENT_TABLES = [
    "possessions", "chances", "chance_players", "matchups", "shots", "free_throws", "rebounds",
    "turnovers", "fouls", "timeouts", "passes", "touches", "dribbles", "picks", "handoffs",
    "off_ball_screens", "drives", "isolations", "posts", "closeouts",
]


def load_game_data(game_id):
    with open(MATCH_DIR / str(game_id) / f"{game_id}_game_data.json") as f:
        return json.load(f)


def load_events(game_id):
    """Return dict of DataFrames, one per event table."""
    with open(MATCH_DIR / str(game_id) / f"{game_id}_dynamic_events.json") as f:
        raw = json.load(f)
    return {k: pd.DataFrame(raw.get(k, [])) for k in EVENT_TABLES}


@dataclass
class Tracking:
    """Live frames only (frames with 10 players). Arrays indexed by live-frame row.

    Player slots are sorted by playerId within each team so a slot holds the same player
    across consecutive frames unless a substitution happened (check ids).
    """

    game_id: int
    frame: np.ndarray  # (n,) frameIdx
    wall: np.ndarray  # (n,) ms
    game_clock: np.ndarray
    shot_clock: np.ndarray
    period: np.ndarray
    clock_stopped: np.ndarray
    home_id: np.ndarray  # (n,5)
    home_xy: np.ndarray  # (n,5,2) broadcast frame
    home_speed: np.ndarray
    home_pe: np.ndarray
    home_det: np.ndarray
    away_id: np.ndarray
    away_xy: np.ndarray
    away_speed: np.ndarray
    away_pe: np.ndarray
    away_det: np.ndarray
    ball_xyz: np.ndarray  # (n,3)
    ball_det: np.ndarray
    ball_pe: np.ndarray
    home_team: int
    away_team: int

    def row_of_frame(self):
        """Map frameIdx -> live row (or -1)."""
        m = np.full(self.frame.max() + 2, -1, dtype=np.int64)
        m[self.frame] = np.arange(len(self.frame))
        return m


def _parse_players(players):
    players = sorted(players, key=lambda p: p["playerId"])
    ids = np.array([p["playerId"] for p in players], dtype=np.int64)
    xy = np.array([p["xyz"][:2] for p in players], dtype=np.float32)
    sp = np.array([p["speed"] for p in players], dtype=np.float32)
    pe = np.array([p["predError"] for p in players], dtype=np.float32)
    det = np.array([p["isDetected"] for p in players], dtype=np.int8)
    return ids, xy, sp, pe, det


def build_tracking_cache(game_id):
    path = MATCH_DIR / str(game_id) / f"{game_id}_tracking_data.jsonl.gz"
    cols = {k: [] for k in [
        "frame", "wall", "game_clock", "shot_clock", "period", "clock_stopped",
        "home_id", "home_xy", "home_speed", "home_pe", "home_det",
        "away_id", "away_xy", "away_speed", "away_pe", "away_det",
        "ball_xyz", "ball_det", "ball_pe"]}
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if len(d["homePlayers"]) != 5 or len(d["awayPlayers"]) != 5:
                continue
            cols["frame"].append(d["frameIdx"])
            cols["wall"].append(d["wallClock"])
            cols["game_clock"].append(d["gameClock"])
            cols["shot_clock"].append(np.nan if d["shotClock"] is None else d["shotClock"])
            cols["period"].append(d["period"])
            cols["clock_stopped"].append(bool(d["gameClockStopped"]))
            for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
                ids, xy, sp, pe, det = _parse_players(d[key])
                cols[f"{side}_id"].append(ids)
                cols[f"{side}_xy"].append(xy)
                cols[f"{side}_speed"].append(sp)
                cols[f"{side}_pe"].append(pe)
                cols[f"{side}_det"].append(det)
            b = d["ball"] or {}
            cols["ball_xyz"].append(b.get("xyz", [np.nan] * 3))
            cols["ball_det"].append(float(b.get("isDetected", np.nan)))
            cols["ball_pe"].append(float(b.get("predError", np.nan)))
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    gd = load_game_data(game_id)
    arrays["home_team"] = np.int64(gd["homeTeam"]["teamId"])
    arrays["away_team"] = np.int64(gd["awayTeam"]["teamId"])
    np.savez_compressed(CACHE_DIR / f"tracking_{game_id}.npz", **arrays)
    return arrays


def load_tracking(game_id):
    p = CACHE_DIR / f"tracking_{game_id}.npz"
    if not p.exists():
        build_tracking_cache(game_id)
    z = np.load(p)
    return Tracking(game_id=game_id, **{k: (z[k] if z[k].ndim else z[k].item()) for k in z.files})


@dataclass
class FrameContext:
    """Per live-frame possession context, aligned to Tracking rows."""

    off_is_home: np.ndarray  # (n,) 1 home offense, 0 away offense, -1 no possession
    flip: np.ndarray  # (n,) bool: rotate 180 deg to get attack frame
    possession_idx: np.ndarray  # (n,) index into events['possessions'] or -1
    chance_idx: np.ndarray  # (n,) index into events['chances'] or -1
    handler_id: np.ndarray  # (n,) playerId with the ball (from touches) or -1
    in_frontcourt: np.ndarray  # (n,) bool: chance has reached the frontcourt


def _interval_index(frames, starts, ends):
    """For each frame, index of the interval [start, end) containing it (last wins), else -1."""
    out = np.full(len(frames), -1, dtype=np.int64)
    order = np.argsort(starts)
    for i in order:
        s, e = starts[i], ends[i]
        if e < s:
            continue
        lo, hi = np.searchsorted(frames, s), np.searchsorted(frames, e, side="right")
        out[lo:hi] = i
    return out


def build_frame_context(tr: Tracking, ev):
    poss = ev["possessions"]
    ch = ev["chances"]
    pidx = _interval_index(tr.frame, poss["startFrame"].values, poss["endFrame"].values)
    cidx = _interval_index(tr.frame, ch["startFrame"].values, ch["endFrame"].values)
    off_is_home = np.full(len(tr.frame), -1, dtype=np.int8)
    flip = np.zeros(len(tr.frame), dtype=bool)
    ok = pidx >= 0
    off_team = poss["offTeamId"].values[pidx[ok]]
    off_is_home[ok] = (off_team == tr.home_team).astype(np.int8)
    flip[ok] = ~poss["leftHoop"].values[pidx[ok]].astype(bool)
    handler = np.full(len(tr.frame), -1, dtype=np.int64)
    t = ev["touches"]
    tidx = _interval_index(tr.frame, t["startFrame"].values, t["endFrame"].values)
    handler[tidx >= 0] = t["playerId"].values[tidx[tidx >= 0]]
    fc = np.zeros(len(tr.frame), dtype=bool)
    okc = cidx >= 0
    fcf = ch["frontcourtFrame"].values.astype(float)
    fcf_frame = fcf[cidx[okc]]
    fc[okc] = (~np.isnan(fcf_frame)) & (tr.frame[okc] >= np.nan_to_num(fcf_frame, nan=1e12))
    return FrameContext(off_is_home, flip, pidx, cidx, handler, fc)


def attack_frame_positions(tr: Tracking, ctx: FrameContext, rows):
    """Offense/defense positions (attack frame) for the given live rows.

    Returns dict with off_xy, def_xy (m,5,2), off_id, def_id, off_pe, def_pe, off_det, def_det,
    off_speed, def_speed, ball (m,3).
    """
    rows = np.asarray(rows)
    oh = ctx.off_is_home[rows]
    sgn = np.where(ctx.flip[rows], -1.0, 1.0)[:, None, None]
    home = dict(xy=tr.home_xy[rows], id=tr.home_id[rows], pe=tr.home_pe[rows], det=tr.home_det[rows], speed=tr.home_speed[rows])
    away = dict(xy=tr.away_xy[rows], id=tr.away_id[rows], pe=tr.away_pe[rows], det=tr.away_det[rows], speed=tr.away_speed[rows])
    is_home = (oh == 1)[:, None]
    out = {}
    for k in ("xy", "id", "pe", "det", "speed"):
        h, a = home[k], away[k]
        sel = is_home[..., None] if h.ndim == 3 else is_home
        out[f"off_{k}"] = np.where(sel, h, a)
        out[f"def_{k}"] = np.where(sel, a, h)
    out["off_xy"] = out["off_xy"] * sgn
    out["def_xy"] = out["def_xy"] * sgn
    ball = tr.ball_xyz[rows].copy()
    ball[:, :2] *= sgn[:, :, 0]
    out["ball"] = ball
    out["valid"] = oh >= 0
    return out


def player_velocity(tr: Tracking, rows, side, lag_frames=5):
    """Finite-difference velocity (ft/s) in the *broadcast* frame for a team's slots.

    Uses a centred window of `lag_frames` on each side, requiring the same playerId in the
    slot at both ends; otherwise falls back to a one-sided difference or zero.
    """
    rows = np.asarray(rows)
    xy = getattr(tr, f"{side}_xy")
    ids = getattr(tr, f"{side}_id")
    n = len(tr.frame)
    r0 = np.clip(rows - lag_frames, 0, n - 1)
    r1 = np.clip(rows + lag_frames, 0, n - 1)
    # frames must be contiguous (no gap) and same player
    good = (tr.frame[r1] - tr.frame[r0] == (r1 - r0)) & (r1 > r0)
    same = (ids[r0] == ids[rows]) & (ids[r1] == ids[rows]) & good[:, None]
    dt = ((tr.frame[r1] - tr.frame[r0]) / 25.0)[:, None, None]
    v = (xy[r1] - xy[r0]) / np.maximum(dt, 1e-6)
    v = np.where(same[..., None], v, 0.0)
    return v.astype(np.float32)


def attack_frame_velocity(tr: Tracking, ctx: FrameContext, rows, lag_frames=5):
    rows = np.asarray(rows)
    vh = player_velocity(tr, rows, "home", lag_frames)
    va = player_velocity(tr, rows, "away", lag_frames)
    oh = ctx.off_is_home[rows]
    sgn = np.where(ctx.flip[rows], -1.0, 1.0)[:, None, None]
    is_home = (oh == 1)[:, None, None]
    return np.where(is_home, vh, va) * sgn, np.where(is_home, va, vh) * sgn
