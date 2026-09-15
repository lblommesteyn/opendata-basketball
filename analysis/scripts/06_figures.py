"""Stage 6: summary figures for the paper (models, value maps, event curves, validation, robustness)."""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spaceholes.config import CACHE_DIR, FIG_DIR, FPS, GAME_IDS, OUT_DIR  # noqa: E402
from spaceholes.control import control_kinematic, control_probabilistic, control_voronoi  # noqa: E402
from spaceholes.io import attack_frame_positions, attack_frame_velocity, build_frame_context, load_events, load_tracking  # noqa: E402
from spaceholes.value import value_geometric  # noqa: E402
from spaceholes.viz import plot_field, plot_players  # noqa: E402

MAIN = "prob_emp_A_tau0.5"
KIND_ORDER = ["pick", "handoff", "off_ball_screen", "drive", "isolation", "pass", "closeout", "shot"]
COLORS = {"pick": "#7b2cbf", "handoff": "#9d4edd", "off_ball_screen": "#c77dff", "drive": "#e63946", "isolation": "#f4a261",
          "pass": "#2a9d8f", "closeout": "#264653", "shot": "#e9c46a"}


def fig_models_and_values():
    g = GAME_IDS[0]
    tr, ev = load_tracking(g), load_events(g)
    ctx = build_frame_context(tr, ev)
    pf = pd.read_parquet(OUT_DIR / f"frames_{g}.parquet")
    # a frame with a known handler and mid-range A
    cand = pf[(pf.handler_id > 0) & (pf.shot_clock.between(6, 16))]
    r = int(cand.iloc[len(cand) // 3]["row"])
    pos = attack_frame_positions(tr, ctx, [r])
    ov, dv = attack_frame_velocity(tr, ctx, [r])
    V = np.load(CACHE_DIR / "value_maps_loo.npz")[str(g)]
    C = {
        "Euclidean Voronoi": control_voronoi(pos["off_xy"], pos["def_xy"])[0],
        "Velocity-aware time-to-arrival": control_kinematic(pos["off_xy"], ov, pos["def_xy"], dv)[0],
        "Uncertainty-aware probabilistic": control_probabilistic(pos["off_xy"], ov, pos["off_pe"], pos["off_det"], pos["def_xy"], dv, pos["def_pe"], pos["def_det"])[0],
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (name, c) in zip(axes[0], C.items()):
        plot_field(ax, c, vmax=1, cmap="RdBu")
        h = np.flatnonzero(pos["off_id"][0] == ctx.handler_id[r])
        plot_players(ax, pos["off_xy"][0], pos["def_xy"][0], pos["ball"][0], ov[0], dv[0], h[0] if len(h) else -1)
        ax.set_title(f"C_O: {name}\n(blue = offense first; area = {c.sum():.0f} ft²)", fontsize=10)
    plot_field(axes[1, 0], value_geometric(), vmax=1.5, cmap="viridis")
    axes[1, 0].set_title("V(q): parametric open-shot value", fontsize=10)
    plot_field(axes[1, 1], V, vmax=1.5, cmap="viridis")
    axes[1, 1].set_title("V(q): empirical (other 9 games), shrunk to parametric", fontsize=10)
    H = C["Uncertainty-aware probabilistic"] * V
    plot_field(axes[1, 2], H, vmax=1.2)
    plot_players(axes[1, 2], pos["off_xy"][0], pos["def_xy"][0], pos["ball"][0])
    axes[1, 2].set_title(f"H = C_O · V   (A_τ=0.5 = {np.maximum(H - 0.5, 0).sum():.0f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_models_and_values.png", dpi=130)
    plt.close(fig)


def fig_event_curves():
    z = np.load(OUT_DIR / "event_curves.npz", allow_pickle=True)
    offs, kind = z["offs"], z["kind"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, metric, title in zip(axes, [MAIN, "vor_emp_A_tau0.5", "handler_ndd"],
                                 ["Probabilistic H: A(t), τ=0.5", "Voronoi H: A(t), τ=0.5", "Ball-handler nearest-defender distance (ft)"]):
        M = z[metric]
        for k in KIND_ORDER:
            m = kind == k
            if m.sum() < 20:
                continue
            mu = np.nanmean(M[m], 0)
            base = np.nanmean(mu[offs <= -1.0])
            ax.plot(offs, mu - base, color=COLORS[k], lw=2, label=f"{k} (n={m.sum()})")
        ax.axvline(0, color="0.5", lw=0.8)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel("seconds from event start")
        ax.set_ylabel("change vs. pre-event baseline")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_event_curves.png", dpi=130)
    plt.close(fig)


def fig_event_deltas():
    d = pd.read_parquet(OUT_DIR / "event_deltas.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    kinds = [k for k in KIND_ORDER if (d.kind == k).sum() >= 20]
    for ax, col, title in zip(axes, [f"{MAIN}__delta", f"{MAIN}__gain", f"{MAIN}__half_life"],
                              ["ΔA = A(+1.5 s) − A(−1 s)", "Gain to post-event peak", "Advantage half-life (s)"]):
        data = []
        for k in kinds:
            v = d.loc[d.kind == k, col].replace(np.inf, np.nan).dropna().values
            data.append(v)
        bp = ax.boxplot(data, labels=kinds, showfliers=False, patch_artist=True)
        for patch, k in zip(bp["boxes"], kinds):
            patch.set_facecolor(COLORS[k])
            patch.set_alpha(0.6)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_event_deltas.png", dpi=130)
    plt.close(fig)


def fig_validation():
    r = pd.read_csv(OUT_DIR / "validation_results.csv")
    r = r[r.model == "gbm"]
    order = ["ball_only", "baseline", "baseline+voronoi", "baseline+kinematic", "baseline+prob", "baseline+prob+dyn", "prob_only"]
    clf = r[r.auc.notna()].pivot(index="target", columns="features", values="auc")[order]
    reg = r[r.r2.notna()].pivot(index="target", columns="features", values="r2")[order]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [3, 1.2]})
    x = np.arange(len(clf))
    w = 0.8 / len(order)
    cmap = plt.get_cmap("viridis")
    for i, f in enumerate(order):
        axes[0].bar(x + i * w, clf[f], w, label=f, color=cmap(i / (len(order) - 1)))
        axes[1].bar(np.arange(len(reg)) + i * w, reg[f], w, color=cmap(i / (len(order) - 1)))
    axes[0].set_xticks(x + 0.4)
    axes[0].set_xticklabels(clf.index, rotation=20, fontsize=8)
    lo = float(np.nanmin(clf.values))
    axes[0].set_ylim(max(0.5, lo - 0.05), min(1.0, float(np.nanmax(clf.values)) + 0.03))
    axes[0].set_ylabel("AUC (game-held-out)")
    axes[0].set_title("Predicting the next 3 s (gradient boosting, leave-one-game-out)", fontsize=10)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xticks(np.arange(len(reg)) + 0.4)
    axes[1].set_xticklabels(reg.index, rotation=20, fontsize=8)
    axes[1].set_ylabel("R² (game-held-out)")
    axes[1].axhline(0, color="0.5", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_validation.png", dpi=130)
    plt.close(fig)


def fig_robustness():
    t = pd.read_csv(OUT_DIR / "robustness_summary.csv")
    t = t[t.variant != "reference"].sort_values("r_delta")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t["r_A_pre"], 0.38, label="corr(A level) with reference", color="#7b2cbf")
    ax.barh(y + 0.2, t["r_delta"], 0.38, label="corr(ΔA per event) with reference", color="#2a9d8f")
    ax.scatter(t["spearman_kind_ranking"], y, marker="D", color="#e63946", zorder=5, label="Spearman: event-kind ranking")
    ax.set_yticks(y)
    ax.set_yticklabels(t["variant"], fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.axvline(0.9, color="0.6", lw=0.8, ls="--")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Robustness of the measure to modelling choices", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_robustness.png", dpi=130)
    plt.close(fig)


def fig_model_agreement():
    """Frame-level relationship between the three models' advantage measures."""
    pf = pd.concat([pd.read_parquet(OUT_DIR / f"frames_{g}.parquet") for g in GAME_IDS], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    s = pf.sample(min(20000, len(pf)), random_state=0)
    axes[0].hexbin(s["vor_emp_A_tau0.5"], s[MAIN], gridsize=40, bins="log", cmap="magma")
    axes[0].set_xlabel("Voronoi A (τ=0.5)")
    axes[0].set_ylabel("Probabilistic A (τ=0.5)")
    axes[0].set_title(f"r = {pf['vor_emp_A_tau0.5'].corr(pf[MAIN]):.2f}", fontsize=10)
    axes[1].hexbin(s["handler_ndd"], s[MAIN], gridsize=40, bins="log", cmap="magma")
    axes[1].set_xlabel("ball-handler nearest-defender distance (ft)")
    axes[1].set_title(f"r = {pf['handler_ndd'].corr(pf[MAIN]):.2f}", fontsize=10)
    axes[2].hexbin(s["n_extrap"], s[MAIN], gridsize=(10, 40), bins="log", cmap="magma")
    axes[2].set_xlabel("number of extrapolated (undetected) players")
    axes[2].set_title(f"r = {pf['n_extrap'].corr(pf[MAIN]):.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_model_agreement.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["models", "curves", "deltas", "validation", "robustness", "agreement"]
    fns = dict(models=fig_models_and_values, curves=fig_event_curves, deltas=fig_event_deltas, validation=fig_validation,
               robustness=fig_robustness, agreement=fig_model_agreement)
    for w in which:
        fns[w]()
        print("done", w, flush=True)
