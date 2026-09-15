"""Stage 2: changes in advantage around events, event-type curves, half-lives.

Reads outputs/holes/frames_{game}.parquet; writes outputs/holes/event_deltas.parquet,
outputs/holes/event_curves.npz and a text summary.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spaceholes.config import GAME_IDS, OUT_DIR  # noqa: E402
from spaceholes.events import collect_events, event_curves, event_deltas  # noqa: E402
from spaceholes.io import load_events  # noqa: E402
from spaceholes.pipeline import low_confidence  # noqa: E402

METRICS = [
    "prob_emp_A_tau0.5", "prob_emp_A_hole", "prob_emp_A_occ", "prob_emp_A_hole_rim", "prob_emp_A_tau0.7", "prob_emp_A_tau0.3", "prob_emp_A_total", "prob_emp_A_max", "prob_emp_A_cc", "prob_emp_A_rim",
    "prob_emp_A_corner", "prob_empball_A_tau0.5", "prob_geom_A_tau0.5", "prob_unif_A_total",
    "kin_emp_A_tau0.5", "vor_emp_A_tau0.5", "vor_unif_A_total", "handler_ndd", "ndd_mean", "off_hull", "n_extrap", "off_pe_mean", "def_pe_mean",
]


HORIZON = 3.0  # seconds after the event over which the peak / half-life are measured


def null_events(pf, events, g, n=150, seed=0):
    """Random live half-court frames at least 1 s away from any marked event: the no-action control."""
    rng = np.random.default_rng(seed + g)
    ev_frames = np.sort(events["frame"].values)
    cand = pf[(pf.shot_clock.notna())].copy()
    idx = np.searchsorted(ev_frames, cand["frame"].values)
    prev = ev_frames[np.clip(idx - 1, 0, len(ev_frames) - 1)]
    nxt = ev_frames[np.clip(idx, 0, len(ev_frames) - 1)]
    far = (np.abs(cand["frame"].values - prev) > 25) & (np.abs(cand["frame"].values - nxt) > 25)
    cand = cand[far]
    take = cand.sample(min(n, len(cand)), random_state=int(rng.integers(1e9)))
    return pd.DataFrame({"game_id": g, "kind": "null", "event_id": [f"null-{g}-{i}" for i in range(len(take))],
                         "frame": take["frame"].values, "chanceId": None, "chance_idx": take["chance_idx"].values})


def main(games=None):
    games = games or GAME_IDS
    all_deltas = []
    curves = {}
    for g in games:
        pf = pd.read_parquet(OUT_DIR / f"frames_{g}.parquet")
        pf = pf[~low_confidence(pf)]
        ev = load_events(g)
        events = collect_events(ev, g)
        ch = ev["chances"].reset_index()
        cmap = dict(zip(ch["id"], ch["index"]))
        events["chance_idx"] = events["chanceId"].map(cmap).fillna(-1).astype(int)
        # keep only events inside sampled half-court frames
        events = events[events["chance_idx"].isin(pf["chance_idx"].unique())].reset_index(drop=True)
        events = pd.concat([events, null_events(pf, events, g)], ignore_index=True)
        cols = {}
        for m in METRICS:
            r = event_deltas(pf, events, m, horizon=HORIZON)
            for c in ("pre", "post", "delta", "peak", "t_peak", "half_life", "gain"):
                cols[f"{m}__{c}"] = r[c].values
            cols[f"{m}__n_valid"] = r["n_valid"].values
            offs, M = event_curves(pf, events, m)
            curves.setdefault(m, []).append(M)
        d = pd.concat([events, pd.DataFrame(cols, index=events.index)], axis=1)
        all_deltas.append(d)
        print(g, len(events), flush=True)
    deltas = pd.concat(all_deltas, ignore_index=True)
    deltas.to_parquet(OUT_DIR / "event_deltas.parquet", index=False)
    np.savez_compressed(OUT_DIR / "event_curves.npz", offs=offs, kind=deltas["kind"].values,
                        **{m: np.vstack(v) for m, v in curves.items()})

    lines = []
    main_m = "prob_emp_A_tau0.5"
    for kind, grp in deltas.groupby("kind"):
        dv = grp[f"{main_m}__delta"].dropna()
        gain = grp[f"{main_m}__gain"].dropna()
        hl_all = grp[f"{main_m}__half_life"].dropna()
        hl = hl_all.replace(np.inf, np.nan).dropna()
        lines.append(f"{kind:16s} n={len(grp):5d}  dA(+1.5s) mean={dv.mean():7.2f} median={dv.median():7.2f} "
                     f"P(dA>0)={np.mean(dv > 0):.2f}  gain-to-peak={gain.mean():6.2f}  half-life median={hl.median():.2f}s "
                     f"repaired-within-{HORIZON:.0f}s={len(hl) / max(len(hl_all), 1):.2f} (n={len(hl_all)})")
    (OUT_DIR / "event_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or None)
