"""Stage 7: action values in the common geometric currency, and checks against SkillCorner labels.

Uses outputs/holes/event_deltas.parquet. Reports, for the main measure:
  * mean delta-A per event kind with bootstrap CIs, relative to the no-action null;
  * splits by SkillCorner context labels (pick coverage, blow-by, closeout action, pass distance);
  * whether the advantage created by an action relates to the chance outcome.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spaceholes.config import GAME_IDS, OUT_DIR  # noqa: E402
from spaceholes.io import load_events  # noqa: E402

MAIN = "prob_emp_A_tau0.5"
RNG = np.random.default_rng(0)


def boot_ci(x, n=2000):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    means = np.array([RNG.choice(x, len(x)).mean() for _ in range(n)])
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def fmt_row(name, x, null_mean):
    m, lo, hi = boot_ci(x)
    return f"{name:34s} n={np.isfinite(x).sum():5d}  dA={m:7.1f} [{lo:6.1f}, {hi:6.1f}]  excess over null={m - null_mean:6.1f}"


def main():
    d = pd.read_parquet(OUT_DIR / "event_deltas.parquet")
    delta, gain, pre, hl = f"{MAIN}__delta", f"{MAIN}__gain", f"{MAIN}__pre", f"{MAIN}__half_life"
    d = d[d[f"{MAIN}__n_valid"] >= 8]
    null = d[d.kind == "null"]
    null_mean = null[delta].mean()
    null_gain = null[gain].mean()
    L = ["== Action values: mean dA (A(+1.5s) - A(-1s)) in ft^2*pts, bootstrap 95% CI ==",
         f"null (no action) mean dA = {null_mean:.1f}, mean gain-to-peak = {null_gain:.1f}", ""]
    for kind, grp in d.groupby("kind"):
        L.append(fmt_row(kind, grp[delta].values, null_mean))
    L.append("")
    L.append("== Mann-Whitney vs null (two-sided) ==")
    for kind, grp in d.groupby("kind"):
        if kind == "null":
            continue
        p = stats.mannwhitneyu(grp[delta].dropna(), null[delta].dropna()).pvalue
        L.append(f"{kind:20s} p={p:.2e}")
    L.append("")
    L.append("== Splits by SkillCorner context ==")
    pk = d[d.kind == "pick"]
    for col in ("bhrDefType", "scrDefType", "locationType", "direct"):
        for v, grp in pk.groupby(col):
            L.append(fmt_row(f"pick {col}={v}", grp[delta].values, null_mean))
    dr = d[d.kind == "drive"]
    for col in ("blowby", "endType", "category"):
        for v, grp in dr.groupby(col):
            L.append(fmt_row(f"drive {col}={v}", grp[delta].values, null_mean))
    co = d[d.kind == "closeout"]
    for v, grp in co.groupby("bhrAction"):
        L.append(fmt_row(f"closeout bhrAction={v}", grp[delta].values, null_mean))
    ob = d[d.kind == "off_ball_screen"]
    for col in ("cutterDefType", "screenerDefType", "ledToShot"):
        for v, grp in ob.groupby(col):
            L.append(fmt_row(f"off_ball_screen {col}={v}", grp[delta].values, null_mean))
    ps = d[d.kind == "pass"].copy()
    ps["dist_bin"] = pd.cut(ps["distance"].astype(float), [0, 12, 20, 30, 100], labels=["<12ft", "12-20", "20-30", ">30 (skip)"])
    for v, grp in ps.groupby("dist_bin", observed=True):
        L.append(fmt_row(f"pass distance {v}", grp[delta].values, null_mean))
    for v, grp in ps.groupby("assistOpp"):
        L.append(fmt_row(f"pass assistOpp={v}", grp[delta].values, null_mean))
    L.append("")
    L.append("== Pre-shot advantage vs shot labels (A at -1..0 s before release) ==")
    sh = d[d.kind == "shot"]
    for col in ("contestLevel", "catchAndShoot", "outcome"):
        for v, grp in sh.groupby(col):
            m, lo, hi = boot_ci(grp[pre].values)
            L.append(f"shot {col}={v!s:10s} n={len(grp):4d}  pre-A={m:6.1f} [{lo:6.1f},{hi:6.1f}]")
    sq = sh[["shotQuality", pre]].dropna().astype(float)
    if len(sq) > 10:
        r = stats.spearmanr(sq["shotQuality"], sq[pre])
        L.append(f"Spearman(pre-shot A, SkillCorner shotQuality) = {r.correlation:.3f} (p={r.pvalue:.1e}, n={len(sq)})")
    cd = sh[["closestDefDist", pre]].dropna().astype(float)
    r = stats.spearmanr(cd["closestDefDist"], cd[pre])
    L.append(f"Spearman(pre-shot A, closest defender distance) = {r.correlation:.3f} (n={len(cd)})")
    L.append("")
    L.append("== Half-life / repair by kind (finite = repaired within 3 s) ==")
    for kind, grp in d.groupby("kind"):
        h = grp[hl].dropna()
        fin = h.replace(np.inf, np.nan).dropna()
        L.append(f"{kind:20s} n={len(h):4d}  repaired within 3s: {len(fin) / max(len(h), 1):.2f}   median half-life (repaired) = {fin.median():.2f}s")
    L.append("")
    L.append("== Does the advantage an action creates convert? (chance points by dA tercile, per kind) ==")
    ch = {}
    for g in GAME_IDS:
        c = load_events(g)["chances"]
        ch.update({(g, i): (p, u) for i, (p, u) in enumerate(zip(c["ptsScored"], c["usable"]))})
    d["chance_pts"] = [ch.get((g, i), (np.nan, None))[0] for g, i in zip(d.game_id, d.chance_idx)]
    for kind in ("pick", "off_ball_screen", "drive", "pass", "handoff"):
        grp = d[(d.kind == kind)].dropna(subset=[delta, "chance_pts"])
        if len(grp) < 60:
            continue
        grp = grp.assign(terc=pd.qcut(grp[delta], 3, labels=["low", "mid", "high"]))
        pts = grp.groupby("terc", observed=True)["chance_pts"].mean()
        r = stats.spearmanr(grp[delta], grp["chance_pts"])
        L.append(f"{kind:16s} pts/chance by dA tercile: low={pts['low']:.2f} mid={pts['mid']:.2f} high={pts['high']:.2f}   spearman={r.correlation:.3f} p={r.pvalue:.2e} n={len(grp)}")
    (OUT_DIR / "action_values.txt").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
