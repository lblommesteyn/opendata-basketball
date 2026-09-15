"""Stage 4: sensitivity of the advantage measure and of event conclusions to modelling choices.

For a fixed subsample of event windows (all event kinds, several games) we recompute the field
under perturbations and report (a) the correlation of per-frame A with the reference run and
(b) the stability of the event-kind ranking by mean delta-A (Spearman) and of per-event delta-A.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spaceholes import config  # noqa: E402
from spaceholes.config import CACHE_DIR, CELL_AREA, FPS, GAME_IDS, GRID_Q, OUT_DIR  # noqa: E402
from spaceholes.control import control_kinematic, control_probabilistic, control_voronoi  # noqa: E402
from spaceholes.events import collect_events  # noqa: E402
from spaceholes.field import summaries  # noqa: E402
from spaceholes.io import attack_frame_positions, attack_frame_velocity, build_frame_context, load_events, load_tracking  # noqa: E402
from spaceholes.value import value_geometric  # noqa: E402

RNG = np.random.default_rng(42)
GAMES = GAME_IDS[:4]
N_EVENTS_PER_KIND = 40
PRE, POST = 1.0, 1.5  # seconds


def sample_windows():
    """Pick events and the frames (pre-window and post-window) needed for delta-A."""
    win = []
    for g in GAMES:
        tr = load_tracking(g)
        ev = load_events(g)
        ctx = build_frame_context(tr, ev)
        rowmap = tr.row_of_frame()
        events = collect_events(ev, g)
        ch = ev["chances"].reset_index()
        cmap = dict(zip(ch["id"], ch["index"]))
        events["chance_idx"] = events["chanceId"].map(cmap).fillna(-1).astype(int)
        for kind, grp in events.groupby("kind"):
            take = grp.sample(min(N_EVENTS_PER_KIND, len(grp)), random_state=1)
            for _, e in take.iterrows():
                f0 = int(e["frame"])
                pre_f = np.arange(f0 - int(PRE * FPS), f0 + 1, 5)
                post_f = np.arange(f0 + 5, f0 + int(POST * FPS) + 1, 5)
                pr = rowmap[np.clip(pre_f, 0, len(rowmap) - 1)]
                po = rowmap[np.clip(post_f, 0, len(rowmap) - 1)]
                if (pr < 0).any() or (po < 0).any():
                    continue
                if not (ctx.chance_idx[pr] == e["chance_idx"]).all() or not (ctx.chance_idx[po] == e["chance_idx"]).all():
                    continue
                win.append(dict(game=g, kind=kind, event_id=e["event_id"], pre_rows=pr, post_rows=po))
    return win


def field_summary(tr, ctx, rows, V, model="prob", sigma_kw=None, perturb=False, grid=GRID_Q, cell_area=CELL_AREA, smooth_lag=5, tau=0.5, motion_kw=None):
    pos = attack_frame_positions(tr, ctx, rows)
    off_v, def_v = attack_frame_velocity(tr, ctx, rows, lag_frames=smooth_lag)
    off, de = pos["off_xy"].copy(), pos["def_xy"].copy()
    if perturb:
        # draw positions from N(x, (predError/1.645)^2): predError is a 90% bound
        off += RNG.normal(size=off.shape) * (pos["off_pe"] / 1.645)[..., None]
        de += RNG.normal(size=de.shape) * (pos["def_pe"] / 1.645)[..., None]
    motion_kw = motion_kw or {}
    if model == "vor":
        c = control_voronoi(off, de, grid)
    elif model == "kin":
        c = control_kinematic(off, off_v, de, def_v, grid, **motion_kw)
    else:
        c = control_probabilistic(off, off_v, pos["off_pe"], pos["off_det"], de, def_v, pos["def_pe"], pos["def_det"], grid,
                                  motion_kw=motion_kw, **(sigma_kw or {}))
    h = c * V[None, :]
    return np.maximum(h - tau, 0).sum(1) * cell_area, h.sum(1) * cell_area


def run_variant(win, name, value="emp", V=None, **kw):
    """Return per-window (pre, post) means under one configuration."""
    vm = np.load(CACHE_DIR / "value_maps_loo.npz")
    out = []
    cache = {}
    for w in win:
        g = w["game"]
        if g not in cache:
            tr = load_tracking(g)
            ev = load_events(g)
            cache[g] = (tr, build_frame_context(tr, ev))
        tr, ctx = cache[g]
        if V is not None:
            Vg = V
        elif value == "emp":
            Vg = vm[str(g)]
        else:
            Vg = value_geometric(kw.get("grid", GRID_Q))
        a_pre, tot_pre = field_summary(tr, ctx, w["pre_rows"], Vg, **kw)
        a_post, tot_post = field_summary(tr, ctx, w["post_rows"], Vg, **kw)
        out.append(dict(variant=name, game=g, kind=w["kind"], event_id=w["event_id"], pre=a_pre.mean(), post=a_post.mean(),
                        pre_tot=tot_pre.mean(), post_tot=tot_post.mean()))
    d = pd.DataFrame(out)
    d["delta"] = d["post"] - d["pre"]
    return d


def coarse_grid(res):
    gx = np.arange(-config.HALF_LENGTH + res / 2, 0.0, res)
    gy = np.arange(-config.HALF_WIDTH + res / 2, config.HALF_WIDTH, res)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


def main():
    win = sample_windows()
    print("windows", len(win), pd.Series([w["kind"] for w in win]).value_counts().to_dict(), flush=True)
    variants = {
        "reference": dict(),
        "perturb_positions_1": dict(perturb=True),
        "perturb_positions_2": dict(perturb=True),
        "perturb_positions_3": dict(perturb=True),
        "sigma0_x2": dict(sigma_kw=dict(sigma0=0.30)),
        "sigma0_half": dict(sigma_kw=dict(sigma0=0.075)),
        "kappa_0": dict(sigma_kw=dict(kappa=0.0)),
        "kappa_0.2": dict(sigma_kw=dict(kappa=0.2)),
        "no_extrap_inflation": dict(sigma_kw=dict(inflate=1.0)),
        "extrap_inflate_3": dict(sigma_kw=dict(inflate=3.0)),
        "vmax_17": dict(motion_kw=dict(v_max=17.0)),
        "vmax_23": dict(motion_kw=dict(v_max=23.0)),
        "amax_14": dict(motion_kw=dict(a_max=14.0)),
        "amax_26": dict(motion_kw=dict(a_max=26.0)),
        "reaction_0": dict(motion_kw=dict(reaction=0.0)),
        "reaction_0.4": dict(motion_kw=dict(reaction=0.4)),
        "smooth_lag_2": dict(smooth_lag=2),
        "smooth_lag_10": dict(smooth_lag=10),
        "tau_0.3": dict(tau=0.3),
        "tau_0.7": dict(tau=0.7),
        "value_geom": dict(value="geom"),
        "model_kin": dict(model="kin"),
        "model_vor": dict(model="vor"),
    }
    # grid resolution variants need a matching value map: use the geometric value for both
    g2 = coarse_grid(2.0)
    variants["grid_2ft_geom"] = dict(grid=g2, cell_area=4.0, V=value_geometric(g2))
    variants["grid_1ft_geom"] = dict(value="geom")
    res = []
    for name, kw in variants.items():
        d = run_variant(win, name, **dict(kw))
        res.append(d)
        print(name, "mean delta by kind:", d.groupby("kind")["delta"].mean().round(2).to_dict(), flush=True)
    res = pd.concat(res, ignore_index=True)
    res.to_parquet(OUT_DIR / "robustness_runs.parquet", index=False)
    ref = res[res.variant == "reference"].set_index("event_id")
    rows = []
    kinds_ref = ref.groupby("kind")["delta"].mean()
    for name, d in res.groupby("variant"):
        d = d.set_index("event_id").loc[ref.index]
        kinds = d.groupby("kind")["delta"].mean().loc[kinds_ref.index]
        rows.append(dict(variant=name,
                         r_A_pre=np.corrcoef(ref["pre"], d["pre"])[0, 1],
                         r_delta=np.corrcoef(ref["delta"], d["delta"])[0, 1],
                         spearman_kind_ranking=spearmanr(kinds_ref, kinds).correlation,
                         sign_agreement=np.mean(np.sign(ref["delta"]) == np.sign(d["delta"])),
                         mean_A_pre=d["pre"].mean()))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_DIR / "robustness_summary.csv", index=False)
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
