"""Stage 3: does the exploitable-space field predict what happens next, beyond baselines?

Per sampled frame (5 Hz, half-court, live) we predict events in the next `H_SEC` seconds of the
same chance, with game-held-out cross-validation (GroupKFold by game). Feature sets are nested
so that the incremental value of the geometric field can be read off directly.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spaceholes.config import FPS, FT_LINE_X, GAME_IDS, OUT_DIR, PAINT_HALF_WIDTH  # noqa: E402
from spaceholes.io import load_events  # noqa: E402
from spaceholes.pipeline import low_confidence  # noqa: E402

H_SEC = 3.0

BASE = ["ball_x", "ball_y", "ball_rim_dist", "handler_rim_dist", "handler_ndd", "ndd_mean", "ndd_max", "n_open6",
        "off_hull", "def_hull", "off_spread_y", "off_depth_x", "def_mean_rim", "off_speed_mean", "def_speed_mean", "shot_clock"]
VOR = ["vor_area", "vor_unif_A_total", "vor_emp_A_total", "vor_emp_A_tau0.5", "vor_emp_A_max", "vor_emp_A_rim", "vor_emp_A_three", "vor_emp_A_cc"]
KIN = ["kin_area", "kin_emp_A_total", "kin_emp_A_tau0.5", "kin_emp_A_max", "kin_emp_A_rim", "kin_emp_A_three", "kin_emp_A_cc"]
PROB = ["prob_area", "prob_emp_A_total", "prob_emp_A_tau0.3", "prob_emp_A_tau0.5", "prob_emp_A_tau0.7", "prob_emp_A_max",
        "prob_emp_A_rim", "prob_emp_A_corner", "prob_emp_A_three", "prob_emp_A_paint", "prob_emp_A_cc", "prob_emp_n_holes",
        "prob_empball_A_tau0.5", "prob_empball_A_total", "prob_empball_A_cc", "prob_emp_A_hole", "prob_emp_A_occ", "prob_emp_A_hole_rim"]
DYN = ["dA_tau0.5_1s", "dA_total_1s", "dA_cc_1s", "dA_hole_1s"]  # change over the previous second

FEATURE_SETS = {
    "ball_only": ["ball_x", "ball_y", "ball_rim_dist", "shot_clock"],
    "baseline": BASE,
    "baseline+voronoi": BASE + VOR,
    "baseline+kinematic": BASE + KIN,
    "baseline+prob": BASE + PROB,
    "baseline+prob+dyn": BASE + PROB + DYN,
    "prob_only": PROB,
    "prob+dyn_only": PROB + DYN,
}


def add_dynamics(pf):
    pf = pf.sort_values(["game_id", "frame"]).copy()
    for m, name in (("prob_emp_A_tau0.5", "dA_tau0.5_1s"), ("prob_emp_A_total", "dA_total_1s"), ("prob_emp_A_cc", "dA_cc_1s"), ("prob_emp_A_hole", "dA_hole_1s")):
        prev = pf.groupby(["game_id", "chance_idx"])[m].shift(5)  # 5 samples at 5 Hz = 1 s
        pf[name] = pf[m] - prev
    return pf


def make_targets(pf, g, ev):
    """Future-event targets within H_SEC seconds, same chance."""
    hz = int(H_SEC * FPS)
    ch = ev["chances"].reset_index()
    cmap = dict(zip(ch["id"], ch["index"]))
    frames = pf["frame"].values
    cidx = pf["chance_idx"].values

    def within(table, fcol, mask=None):
        d = ev[table]
        if mask is not None:
            d = d[mask]
        d = d.assign(cidx=d["chanceId"].map(cmap))
        out = np.zeros(len(pf), dtype=bool)
        for f, c in zip(d[fcol].values, d["cidx"].values):
            hit = (frames < f) & (frames >= f - hz) & (cidx == c)
            out |= hit
        return out

    sh = ev["shots"]
    t = {}
    t["shot_3s"] = within("shots", "startFrame")
    t["open_shot_3s"] = within("shots", "startFrame", sh["contestLevel"].isin(["open", "light"]))
    t["rim_shot_3s"] = within("shots", "startFrame", sh["distance"] < 6)
    t["blowby_3s"] = within("drives", "startFrame", ev["drives"]["blowby"].astype(bool))
    t["assist_opp_3s"] = within("passes", "startFrame", ev["passes"]["assistOpp"].astype(bool))
    t["paint_touch_3s"] = within("touches", "startFrame", ev["touches"]["regionsIn"].apply(lambda r: any(x in ("key", "ra") for x in r)))
    # continuous: shot quality of the next shot within window (NaN otherwise)
    sq = np.full(len(pf), np.nan)
    d = sh.assign(cidx=sh["chanceId"].map(cmap)).sort_values("startFrame")
    for f, c, q in zip(d["startFrame"].values, d["cidx"].values, d["shotQuality"].values):
        hit = (frames < f) & (frames >= f - hz) & (cidx == c) & np.isnan(sq)
        sq[hit] = q
    t["next_shot_quality"] = sq
    # chance value (points of the chance) — every frame of the chance shares it
    pts = ch.set_index("index")["ptsScored"]
    t["chance_points"] = pd.Series(cidx).map(pts).values.astype(float)
    return pd.DataFrame(t, index=pf.index)


def cv_eval(X, y, groups, kind, model):
    """Leave-one-game-out. Returns pooled metrics plus the mean/sd of the per-held-out-game metric."""
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    pred = np.full(len(y), np.nan)
    per_fold = []
    for tr, te in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1] if kind == "clf" else model.predict(X[te])
        if kind == "clf" and 0 < y[te].mean() < 1:
            per_fold.append(roc_auc_score(y[te], pred[te]))
        elif kind == "reg":
            per_fold.append(r2_score(y[te], pred[te]))
    per_fold = np.array(per_fold)
    folds = json.dumps([round(float(v), 4) for v in per_fold])
    if kind == "clf":
        return dict(auc=roc_auc_score(y, pred), auc_fold_mean=per_fold.mean(), auc_fold_sd=per_fold.std(ddof=1),
                    logloss=log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6)), n=len(y), rate=float(y.mean()), folds=folds)
    return dict(r2=r2_score(y, pred), r2_fold_mean=per_fold.mean(), r2_fold_sd=per_fold.std(ddof=1), n=len(y), folds=folds)


def main(games=None):
    games = games or GAME_IDS
    frames, targets = [], []
    for g in games:
        pf = pd.read_parquet(OUT_DIR / f"frames_{g}.parquet")
        pf = pf[~low_confidence(pf)].reset_index(drop=True)
        ev = load_events(g)
        frames.append(pf)
        targets.append(make_targets(pf, g, ev))
    pf = add_dynamics(pd.concat(frames, ignore_index=True))
    T = pd.concat(targets, ignore_index=True).loc[pf.index]
    # avoid trivially-late frames: require at least H_SEC of chance remaining? No — keep all, the
    # target is simply "does X happen within 3 s"; frames where the chance ends sooner are negatives.
    results = []
    for tname in ["shot_3s", "open_shot_3s", "rim_shot_3s", "blowby_3s", "assist_opp_3s", "paint_touch_3s", "next_shot_quality", "chance_points"]:
        y = T[tname].values
        ok = np.isfinite(y)
        for fs_name, cols in FEATURE_SETS.items():
            X = pf.loc[ok, cols].values.astype(float)
            X = np.where(np.isfinite(X), X, np.nan)
            grp = pf.loc[ok, "game_id"].values
            if tname in ("next_shot_quality", "chance_points"):
                for mname, model in (("gbm", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15)),
                                     ("linear", make_pipeline(StandardScaler(), Ridge(alpha=1.0)))):
                    Xm = X if mname == "gbm" else np.nan_to_num(X, nan=0.0)
                    r = cv_eval(Xm, y[ok], grp, "reg", model)
                    results.append(dict(target=tname, features=fs_name, model=mname, **r))
            else:
                for mname, model in (("gbm", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15)),
                                     ("logit", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)))):
                    Xm = X if mname == "gbm" else np.nan_to_num(X, nan=0.0)
                    r = cv_eval(Xm, y[ok].astype(int), grp, "clf", model)
                    results.append(dict(target=tname, features=fs_name, model=mname, **r))
            print(tname, fs_name, results[-1], flush=True)
    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / "validation_results.csv", index=False)
    piv = res[res.model.isin(["gbm"])].pivot_table(index="target", columns="features", values=["auc", "r2"])
    print(piv.round(3).to_string())
    T.assign(game_id=pf["game_id"].values, frame=pf["frame"].values).to_parquet(OUT_DIR / "frame_targets.parquet", index=False)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or None)
