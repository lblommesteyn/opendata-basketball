"""Stage 11: does lane openness (widest-path bottleneck to the rim) mean anything?

For every drive: lane openness and the bottleneck location at the drive start (and 0.4 s before),
plus the baselines (handler nearest-defender distance, distance to the rim, A). Tests:
  * does the driver go toward the side of the bottleneck? (direction label vs bottleneck side)
  * blow-by, shot near the basket, kick-out: AUC of openness vs. baselines, leave-one-game-out
  * openness at drive starts vs. at random half-court moments
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("lane", ROOT / "scripts" / "10_lane_width.py")
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)

from spaceholes.config import FPS, GAME_IDS, OUT_DIR, RIM  # noqa: E402
from spaceholes.control import control_probabilistic, nearest_defender_distance  # noqa: E402
from spaceholes.field import to_grid  # noqa: E402
from spaceholes.io import attack_frame_positions, attack_frame_velocity, build_frame_context, load_events, load_tracking  # noqa: E402
from spaceholes.pipeline import low_confidence  # noqa: E402


def lane_features(tr, ctx, rows):
    pos = attack_frame_positions(tr, ctx, rows)
    ov, dv = attack_frame_velocity(tr, ctx, rows)
    C = control_probabilistic(pos["off_xy"], ov, pos["off_pe"], pos["off_det"], pos["def_xy"], dv, pos["def_pe"], pos["def_det"])
    ndd = nearest_defender_distance(pos["off_xy"], pos["def_xy"])
    out = []
    for i in range(len(rows)):
        b = pos["ball"][i][:2]
        val, side, path, bn, xs, ys = lane.widest_path(to_grid(C[i]), b)
        h = np.flatnonzero(pos["off_id"][i] == ctx.handler_id[rows[i]])
        hx, hy = (pos["off_xy"][i][h[0]] if len(h) else b)
        out.append(dict(openness=val, bn_dy=(bn[1] - hy) if bn is not None else np.nan, bn_dist=np.hypot(bn[0] - hx, bn[1] - hy) if bn is not None else np.nan,
                        handler_ndd=ndd[i][h[0]] if len(h) else np.nan, handler_rim=np.hypot(hx - RIM[0], hy - RIM[1]),
                        wall_cells=int(side.sum()), n_extrap=int(10 - pos["off_det"][i].sum() - pos["def_det"][i].sum()),
                        pe_mean=float((pos["off_pe"][i].mean() + pos["def_pe"][i].mean()) / 2)))
    return pd.DataFrame(out)


def main():
    rows_all = []
    for g in GAME_IDS:
        tr, ev = load_tracking(g), load_events(g)
        ctx = build_frame_context(tr, ev)
        rowmap = tr.row_of_frame()
        d = ev["drives"]
        pf = pd.read_parquet(OUT_DIR / f"frames_{g}.parquet")
        pf = pf[~low_confidence(pf)]
        for lag, tag in ((0, "t0"), (10, "tm04")):
            fr = d["startFrame"].values - lag
            r = rowmap[np.clip(fr, 0, len(rowmap) - 1)]
            ok = r >= 0
            f = lane_features(tr, ctx, r[ok])
            f["tag"] = tag
            f["game_id"] = g
            f["drive_id"] = d["id"].values[ok]
            f["blowby"] = d["blowby"].values[ok].astype(bool)
            f["endType"] = d["endType"].values[ok]
            f["direction"] = d["direction"].values[ok]
            f["category"] = d["category"].values[ok]
            f["chanceId"] = d["chanceId"].values[ok]
            rows_all.append(f)
        # null: random half-court frames with a known handler
        s = pf[pf.handler_id > 0].sample(120, random_state=g)
        f = lane_features(tr, ctx, s["row"].values)
        f["tag"] = "null"
        f["game_id"] = g
        rows_all.append(f)
        print(g, flush=True)
    df = pd.concat(rows_all, ignore_index=True)
    ch = pd.concat([load_events(g)["chances"][["id", "ptsScored"]] for g in GAME_IDS])
    df["chance_pts"] = df["chanceId"].map(dict(zip(ch["id"], ch["ptsScored"])))
    df.to_parquet(OUT_DIR / "lane_drives.parquet", index=False)

    L = []
    d0 = df[(df.tag == "t0") & (df.n_extrap < 8) & (df.pe_mean < 4)].copy()
    dm = df[(df.tag == "tm04") & (df.n_extrap < 8) & (df.pe_mean < 4)].copy()
    nl = df[(df.tag == "null") & (df.n_extrap < 8) & (df.pe_mean < 4)]
    L.append(f"drives with lane features at t0: {len(d0)}; null frames: {len(nl)}")
    L.append(f"lane openness: drive start median {d0.openness.median():.3f} (mean {d0.openness.mean():.3f}); 0.4 s before {dm.openness.median():.3f}; random frames {nl.openness.median():.3f} (mean {nl.openness.mean():.3f})")
    L.append(f"  Mann-Whitney drive-start vs null p = {stats.mannwhitneyu(d0.openness.dropna(), nl.openness.dropna()).pvalue:.2e}")
    # direction vs bottleneck side: attack frame has offensive left = +y; SkillCorner 'left'/'right' is the driver's direction
    dd = d0.dropna(subset=["bn_dy"])
    dd = dd[dd.bn_dy.abs() > 1.0]
    toward = ((dd.direction == "left") & (dd.bn_dy > 0)) | ((dd.direction == "right") & (dd.bn_dy < 0))
    L.append(f"driver goes toward the bottleneck side: {toward.mean():.3f} of {len(dd)} drives (binomial p vs 0.5 = {stats.binomtest(int(toward.sum()), len(dd)).pvalue:.2e})")
    ddm = dm.dropna(subset=["bn_dy"]); ddm = ddm[ddm.bn_dy.abs() > 1.0]
    toward_m = ((ddm.direction == "left") & (ddm.bn_dy > 0)) | ((ddm.direction == "right") & (ddm.bn_dy < 0))
    L.append(f"  same, measured 0.4 s BEFORE the drive starts: {toward_m.mean():.3f} of {len(ddm)} (p = {stats.binomtest(int(toward_m.sum()), len(ddm)).pvalue:.2e})")
    # openness by outcome
    for col, val in (("blowby", True),):
        a, b = d0[d0[col] == val].openness, d0[d0[col] != val].openness
        L.append(f"openness | blowby: {a.mean():.3f} (n={len(a)}) vs no blowby {b.mean():.3f} (n={len(b)}), MW p={stats.mannwhitneyu(a.dropna(), b.dropna()).pvalue:.3f}")
    for et, grp in d0.groupby("endType"):
        L.append(f"  endType={et:16s} n={len(grp):4d} openness mean={grp.openness.mean():.3f} median={grp.openness.median():.3f}")
    L.append(f"openness vs points on the chance: spearman={stats.spearmanr(d0.openness, d0.chance_pts, nan_policy='omit').correlation:.3f}")
    # LOGO AUC: openness vs baselines
    def auc(cols, y, dfx):
        X = dfx[cols].values.astype(float); X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        grp = dfx.game_id.values; pred = np.zeros(len(y))
        for tr_, te in GroupKFold(len(np.unique(grp))).split(X, y, grp):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr_], y[tr_])
            pred[te] = m.predict_proba(X[te])[:, 1]
        return roc_auc_score(y, pred)
    for tname, y in (("blow-by", d0.blowby.values.astype(int)), ("shot near basket", (d0.endType == "shotNearBasket").values.astype(int)),
                     ("kick-out", (d0.endType == "kickout").values.astype(int)), ("points > 0 on chance", (d0.chance_pts > 0).values.astype(int))):
        L.append(f"LOGO AUC {tname:22s} rate={y.mean():.2f}  ndd={auc(['handler_ndd'], y, d0):.3f}  ndd+rim={auc(['handler_ndd', 'handler_rim'], y, d0):.3f}  "
                 f"openness={auc(['openness'], y, d0):.3f}  openness+bn_dist={auc(['openness', 'bn_dist'], y, d0):.3f}  all={auc(['handler_ndd', 'handler_rim', 'openness', 'bn_dist'], y, d0):.3f}")
    (OUT_DIR / "lane_validation.txt").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
