"""Align advantage time series to SkillCorner events: delta-A, peaks and half-lives."""

import numpy as np
import pandas as pd

from .config import FPS

# (table, frame column, actor columns, extra columns kept)
EVENT_SPECS = {
    "pick": ("picks", "frame", ["ballhandlerId", "screenerId"], ["bhrDefType", "scrDefType", "direct", "locationType"]),
    "off_ball_screen": ("off_ball_screens", "frame", ["cutterId", "screenerId"], ["cutterDefType", "screenerDefType", "direct", "ledToShot"]),
    "handoff": ("handoffs", "frame", ["receiverId", "setterId"], ["receiverDefType", "setterDefType", "direct"]),
    "drive": ("drives", "startFrame", ["ballhandlerId"], ["category", "blowby", "endType", "direct", "endFrame"]),
    "pass": ("passes", "startFrame", ["passerId", "receiverId"], ["complete", "ledToShot", "assistOpp", "distance", "endFrame"]),
    "closeout": ("closeouts", "startFrame", ["ballhandlerId", "ballhandlerDefId"], ["bhrAction", "startDistance", "endDistance", "endFrame"]),
    "isolation": ("isolations", "startFrame", ["ballhandlerId"], ["direct", "endFrame"]),
    "shot": ("shots", "startFrame", ["shooterId"], ["outcome", "three", "contestLevel", "shotQuality", "closestDefDist", "catchAndShoot", "assisted"]),
}


def collect_events(ev, game_id):
    """Long table of events with a unified `frame`, `kind`, `chanceId`."""
    rows = []
    for kind, (table, fcol, actors, extras) in EVENT_SPECS.items():
        d = ev[table]
        if d.empty:
            continue
        out = pd.DataFrame({"game_id": game_id, "kind": kind, "event_id": d["id"], "frame": d[fcol], "chanceId": d["chanceId"]})
        for i, a in enumerate(actors):
            out[f"actor{i}"] = d[a].values
        for e in extras:
            out[e] = d[e].values if e in d else np.nan
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def series_lookup(df, col):
    """Return (frames, values) sorted, for one game's per-frame table."""
    d = df.sort_values("frame")
    return d["frame"].values, d[col].values, d["chance_idx"].values


def window_values(frames, values, chance_idx, f0, chance, offsets_s):
    """Sample a series at f0 + offsets (seconds) within the same chance; NaN outside."""
    target = f0 + np.round(np.asarray(offsets_s) * FPS).astype(int)
    idx = np.searchsorted(frames, target)
    idx = np.clip(idx, 0, len(frames) - 1)
    ok = (np.abs(frames[idx] - target) <= 3) & (chance_idx[idx] == chance)
    out = np.where(ok, values[idx], np.nan)
    return out


def event_deltas(per_frame, events, metric, pre=1.0, post=1.5, horizon=6.0, step=0.2):
    """For each event compute A before / after, the change, the peak after, and half-life.

    per_frame: DataFrame with frame, chance_idx, metric (one game).
    events: DataFrame with frame, chanceId-> mapped to chance_idx column `chance_idx`.
    """
    frames, vals, cidx = series_lookup(per_frame, metric)
    offs = np.arange(-pre, horizon + 1e-9, step)
    res = []
    for _, e in events.iterrows():
        w = window_values(frames, vals, cidx, int(e["frame"]), e["chance_idx"], offs)
        pre_v = np.nanmean(w[offs <= 0]) if np.isfinite(w[offs <= 0]).any() else np.nan
        post_mask = (offs > 0) & (offs <= post)
        post_v = np.nanmean(w[post_mask]) if np.isfinite(w[post_mask]).any() else np.nan
        after = w[offs > 0]
        peak = np.nanmax(after) if np.isfinite(after).any() else np.nan
        t_peak = offs[offs > 0][np.nanargmax(after)] if np.isfinite(after).any() else np.nan
        # half-life: first time after the peak when the series drops halfway back to pre level
        hl = np.nan
        if np.isfinite(peak) and np.isfinite(pre_v) and peak > pre_v:
            thr = pre_v + 0.5 * (peak - pre_v)
            tt = offs[offs > 0]
            after_peak = (tt > t_peak) & (after <= thr)
            if after_peak.any():
                hl = tt[after_peak][0] - t_peak
            else:
                hl = np.inf  # never decayed within the observed window
        res.append(dict(pre=pre_v, post=post_v, delta=post_v - pre_v, peak=peak, t_peak=t_peak, half_life=hl,
                        gain=peak - pre_v, n_valid=int(np.isfinite(w).sum())))
    out = pd.DataFrame(res, index=events.index)
    return pd.concat([events, out], axis=1)


def event_curves(per_frame, events, metric, pre=2.0, horizon=6.0, step=0.2):
    """Matrix of metric values around events (n_events, n_offsets) and the offsets."""
    frames, vals, cidx = series_lookup(per_frame, metric)
    offs = np.arange(-pre, horizon + 1e-9, step)
    M = np.vstack([window_values(frames, vals, cidx, int(e["frame"]), e["chance_idx"], offs) for _, e in events.iterrows()])
    return offs, M
