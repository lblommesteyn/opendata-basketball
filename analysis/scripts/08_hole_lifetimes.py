"""Stage 8: persistence of holes — track connected high-value regions across time and measure lifetimes.

A hole at a sampled frame is a connected component of {H > tau} that is at least `occupied_radius`
away from every offensive player's centre (so it is space nobody is standing in). Holes are linked
across consecutive 5 Hz samples of the same chance by spatial overlap. We report the lifetime
distribution, the fraction of holes that are 'used' (an offensive player or the ball enters them
before they close), and lifetimes of holes born within 1 s after each event kind vs the null.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spaceholes.config import CACHE_DIR, FPS, GAME_IDS, GRID_Q, OUT_DIR  # noqa: E402
from spaceholes.events import collect_events  # noqa: E402
from spaceholes.field import NX, NY, to_grid  # noqa: E402
from spaceholes.io import attack_frame_positions, build_frame_context, load_events, load_tracking  # noqa: E402
from spaceholes.pipeline import low_confidence  # noqa: E402

TAU = 0.5
MIN_CELLS = 6
OCC_R = 5.0


def hole_labels(h, off_xy):
    dmin = np.linalg.norm(GRID_Q[None, :, :] - off_xy[:, None, :], axis=-1).min(0)
    g = to_grid((h > TAU) & (dmin >= OCC_R))
    lab, n = ndimage.label(g)
    if n:
        sizes = ndimage.sum(np.ones_like(g, dtype=float), lab, index=np.arange(1, n + 1))
        keep = np.flatnonzero(sizes >= MIN_CELLS) + 1
        lab = np.where(np.isin(lab, keep), lab, 0)
    return lab


def track_game(g):
    tr, ev = load_tracking(g), load_events(g)
    ctx = build_frame_context(tr, ev)
    pf = pd.read_parquet(OUT_DIR / f"frames_{g}.parquet").sort_values("frame").reset_index(drop=True)
    C = np.load(CACHE_DIR / f"fields_{g}.npz")["prob"].astype(np.float32)
    V = np.load(CACHE_DIR / "value_maps_loo.npz")[str(g)]
    rows = pf["row"].values
    pos = attack_frame_positions(tr, ctx, rows)
    holes = []  # dict(id, game, chance, birth_frame, death_frame, max_mass, used, birth_x, birth_y)
    active = {}  # label -> hole index
    prev_lab = None
    prev_chance = None
    for i in range(len(pf)):
        h = C[i] * V
        lab = hole_labels(h, pos["off_xy"][i])
        chance = pf.loc[i, "chance_idx"]
        frame = pf.loc[i, "frame"]
        new_active = {}
        if prev_lab is not None and chance == prev_chance and frame - pf.loc[i - 1, "frame"] <= 6:
            for L in np.unique(lab[lab > 0]):
                mask = lab == L
                overlap = prev_lab[mask]
                overlap = overlap[overlap > 0]
                if len(overlap):
                    pl = np.bincount(overlap).argmax()
                    if pl in active and pl not in [v for v in new_active.values()]:
                        new_active[L] = active[pl]
        for L in np.unique(lab[lab > 0]):
            mask = lab == L
            mass = float((np.maximum(to_grid(h) - TAU, 0) * mask).sum())
            cx, cy = ndimage.center_of_mass(mask)
            if L in new_active:
                hidx = new_active[L]
                holes[hidx]["death_frame"] = frame
                holes[hidx]["max_mass"] = max(holes[hidx]["max_mass"], mass)
                holes[hidx]["n_samples"] += 1
            else:
                holes.append(dict(game=g, chance=chance, birth_frame=frame, death_frame=frame, max_mass=mass, n_samples=1,
                                  birth_x=GRID_Q[0, 0] + cx, birth_y=GRID_Q[0, 1] + cy, used=False))
                new_active[L] = len(holes) - 1
            # used: ball enters the hole region (within 2 ft of any cell)
            hidx = new_active[L]
            bx, by = pos["ball"][i, :2]
            gi, gj = int(round(bx - GRID_Q[0, 0])), int(round(by - GRID_Q[0, 1]))
            if 0 <= gi < NX and 0 <= gj < NY and mask[gi, gj]:
                holes[hidx]["used"] = True
        active, prev_lab, prev_chance = new_active, lab, chance
    d = pd.DataFrame(holes)
    d["lifetime_s"] = (d["death_frame"] - d["birth_frame"]) / FPS + 0.2
    return d


def main():
    parts = [track_game(g) for g in GAME_IDS]
    holes = pd.concat(parts, ignore_index=True)
    holes.to_parquet(OUT_DIR / "holes.parquet", index=False)
    L = [f"holes tracked: {len(holes)}  (tau={TAU}, >= {MIN_CELLS} cells, >= {OCC_R} ft from every offensive player)",
         f"lifetime: median {holes.lifetime_s.median():.2f}s, mean {holes.lifetime_s.mean():.2f}s, p90 {holes.lifetime_s.quantile(.9):.2f}s",
         f"fraction used by the ball before closing: {holes.used.mean():.3f}",
         f"used-rate by lifetime: " + ", ".join(f"{lo}-{hi}s: {holes[(holes.lifetime_s >= lo) & (holes.lifetime_s < hi)].used.mean():.2f}" for lo, hi in [(0, 0.6), (0.6, 1.2), (1.2, 2.4), (2.4, 99)]),
         f"used-rate by max mass tercile: " + ", ".join(f"{k}: {v:.2f}" for k, v in holes.groupby(pd.qcut(holes.max_mass, 3, labels=["small", "mid", "large"]), observed=True).used.mean().items()),
         ""]
    # holes born within 1 s after an event, by kind
    rows = []
    for g in GAME_IDS:
        ev = load_events(g)
        events = collect_events(ev, g)
        ch = ev["chances"].reset_index()
        cmap = dict(zip(ch["id"], ch["index"]))
        events["chance_idx"] = events["chanceId"].map(cmap).fillna(-1).astype(int)
        hg = holes[holes.game == g]
        for kind, grp in events.groupby("kind"):
            for f, c in zip(grp["frame"].values, grp["chance_idx"].values):
                born = hg[(hg.chance == c) & (hg.birth_frame > f) & (hg.birth_frame <= f + FPS)]
                rows.append(dict(kind=kind, n_born=len(born), mean_life=born.lifetime_s.mean() if len(born) else np.nan,
                                 max_mass=born.max_mass.max() if len(born) else np.nan, any_used=born.used.any() if len(born) else False))
    # null: random frames
    for g in GAME_IDS:
        pf = pd.read_parquet(OUT_DIR / f"frames_{g}.parquet")
        pf = pf[~low_confidence(pf)]
        hg = holes[holes.game == g]
        rng = np.random.default_rng(g)
        for _, r in pf.sample(150, random_state=int(rng.integers(1e9))).iterrows():
            f, c = r["frame"], r["chance_idx"]
            born = hg[(hg.chance == c) & (hg.birth_frame > f) & (hg.birth_frame <= f + FPS)]
            rows.append(dict(kind="null", n_born=len(born), mean_life=born.lifetime_s.mean() if len(born) else np.nan,
                             max_mass=born.max_mass.max() if len(born) else np.nan, any_used=born.used.any() if len(born) else False))
    e = pd.DataFrame(rows)
    e.to_parquet(OUT_DIR / "holes_by_event.parquet", index=False)
    L.append("holes born within 1 s after the event, by kind:")
    for kind, grp in e.groupby("kind"):
        L.append(f"{kind:16s} n={len(grp):5d}  holes born={grp.n_born.mean():.2f}  mean lifetime={grp.mean_life.mean():.2f}s  "
                 f"max mass={grp.max_mass.mean():.1f}  P(any used)={grp.any_used.mean():.2f}")
    (OUT_DIR / "hole_lifetimes.txt").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
