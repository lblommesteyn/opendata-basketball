"""Per-game computation of control fields, H, advantage summaries and baseline features."""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from .config import CELL_AREA, GRID_Q, OUT_DIR, RIM, SAMPLE_STRIDE
from .control import control_kinematic, control_probabilistic, control_voronoi, nearest_defender_distance
from .field import summaries
from .io import attack_frame_positions, attack_frame_velocity, build_frame_context, load_events, load_tracking
from .value import ball_reach_weight, value_geometric, value_uniform

MODELS = ("vor", "kin", "prob")


def select_rows(tr, ctx, stride=SAMPLE_STRIDE, frontcourt_only=True):
    rows = np.arange(0, len(tr.frame), stride)
    ok = (ctx.off_is_home[rows] >= 0) & (~tr.clock_stopped[rows])
    if frontcourt_only:
        ok &= ctx.in_frontcourt[rows]
    return rows[ok]


def hull_area(xy):
    try:
        return ConvexHull(xy).volume
    except Exception:
        return np.nan


def baseline_features(pos, off_v, def_v, handler_id):
    """Simple spacing / defender-distance features per frame."""
    off, de, ball = pos["off_xy"], pos["def_xy"], pos["ball"]
    m = len(off)
    ndd = nearest_defender_distance(off, de)  # (m,5)
    is_h = pos["off_id"] == handler_id[:, None]
    has_h = is_h.any(1)
    h_ndd = np.where(has_h, (ndd * is_h).sum(1), np.nan)
    hx = np.where(has_h, (off[:, :, 0] * is_h).sum(1), ball[:, 0])
    hy = np.where(has_h, (off[:, :, 1] * is_h).sum(1), ball[:, 1])
    feats = dict(
        ball_x=ball[:, 0], ball_y=ball[:, 1],
        ball_rim_dist=np.hypot(ball[:, 0] - RIM[0], ball[:, 1] - RIM[1]),
        handler_rim_dist=np.hypot(hx - RIM[0], hy - RIM[1]),
        handler_ndd=h_ndd,
        ndd_mean=ndd.mean(1), ndd_max=ndd.max(1), n_open6=(ndd > 6).sum(1),
        off_hull=np.array([hull_area(off[i]) for i in range(m)]),
        def_hull=np.array([hull_area(de[i]) for i in range(m)]),
        off_spread_y=off[:, :, 1].std(1), off_depth_x=off[:, :, 0].std(1),
        def_mean_rim=np.hypot(de[:, :, 0] - RIM[0], de[:, :, 1]).mean(1),
        off_speed_mean=np.linalg.norm(off_v, axis=-1).mean(1),
        def_speed_mean=np.linalg.norm(def_v, axis=-1).mean(1),
        off_pe_mean=pos["off_pe"].mean(1), def_pe_mean=pos["def_pe"].mean(1),
        off_det_frac=pos["off_det"].mean(1), def_det_frac=pos["def_det"].mean(1),
        n_extrap=10 - pos["off_det"].sum(1) - pos["def_det"].sum(1),
    )
    return feats


def compute_game(game_id, value_maps, stride=SAMPLE_STRIDE, batch=64, store_fields=True, rows=None, sigma_kw=None, models=MODELS):
    """Compute per-frame summaries for a game.

    value_maps: dict name -> V (n_cells,) static value maps; 'emp_ball' variant is derived by
    multiplying value_maps['emp'] by the ball-reach weight.
    Returns DataFrame (one row per sampled frame) and dict of stored H fields (float16).
    """
    sigma_kw = sigma_kw or {}
    tr = load_tracking(game_id)
    ev = load_events(game_id)
    ctx = build_frame_context(tr, ev)
    if rows is None:
        rows = select_rows(tr, ctx, stride)
    recs = []
    fields = {m: [] for m in models} if store_fields else None
    for b0 in range(0, len(rows), batch):
        r = rows[b0:b0 + batch]
        pos = attack_frame_positions(tr, ctx, r)
        off_v, def_v = attack_frame_velocity(tr, ctx, r)
        C = {}
        if "vor" in models:
            C["vor"] = control_voronoi(pos["off_xy"], pos["def_xy"])
        if "kin" in models:
            C["kin"] = control_kinematic(pos["off_xy"], off_v, pos["def_xy"], def_v)
        if "prob" in models:
            C["prob"] = control_probabilistic(pos["off_xy"], off_v, pos["off_pe"], pos["off_det"], pos["def_xy"], def_v, pos["def_pe"], pos["def_det"], **sigma_kw)
        reach = ball_reach_weight(GRID_Q, pos["ball"][:, :2])
        rec = dict(game_id=game_id, row=r, frame=tr.frame[r], period=tr.period[r], game_clock=tr.game_clock[r],
                   shot_clock=tr.shot_clock[r], possession_idx=ctx.possession_idx[r], chance_idx=ctx.chance_idx[r],
                   handler_id=ctx.handler_id[r], off_is_home=ctx.off_is_home[r])
        rec.update(baseline_features(pos, off_v, def_v, ctx.handler_id[r]))
        for mname, c in C.items():
            rec[f"{mname}_area"] = c.sum(1) * CELL_AREA
            for vname, V in value_maps.items():
                h = c * V[None, :]
                for k, v in summaries(h, off_xy=pos["off_xy"]).items():
                    rec[f"{mname}_{vname}_{k}"] = v
            h = c * value_maps["emp"][None, :] * reach
            for k, v in summaries(h, off_xy=pos["off_xy"]).items():
                rec[f"{mname}_empball_{k}"] = v
            if store_fields:
                fields[mname].append(c.astype(np.float16))
        recs.append(pd.DataFrame(rec))
    df = pd.concat(recs, ignore_index=True)
    if store_fields:
        fields = {m: np.concatenate(v) for m, v in fields.items()}
    return df, fields


def default_value_maps(shots_other_games=None):
    from .value import fit_empirical_value
    maps = {"geom": value_geometric(), "unif": value_uniform()}
    if shots_other_games is not None:
        maps["emp"], _ = fit_empirical_value(shots_other_games)
    else:
        maps["emp"] = maps["geom"]
    return maps


def low_confidence(df, max_extrap=8, max_pe=4.0):
    """Frames where the tracking is essentially blind (camera cutaway): most players extrapolated
    or the mean expected error is large. C_O collapses to 0.5 there and A to zero, so these frames
    are excluded from event statistics and validation (about 1-2% of half-court frames)."""
    pe = (df["off_pe_mean"] + df["def_pe_mean"]) / 2
    return (df["n_extrap"] >= max_extrap) | (pe > max_pe)
