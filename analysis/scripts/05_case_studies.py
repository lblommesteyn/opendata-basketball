"""Stage 5: interpretable possession-level examples and field visualisations.

Selects events by mechanism (screen opens hole, drive relocates hole, skip pass exploits hole,
defender repairs hole, box-score-productive-but-geometrically-empty, and the reverse) and
renders the H field at several instants, plus the A(t) trace.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spaceholes.config import CACHE_DIR, FIG_DIR, FPS, GRID_Q, GX, GY, OUT_DIR  # noqa: E402
from spaceholes.control import control_probabilistic  # noqa: E402
from spaceholes.field import summaries, to_grid  # noqa: E402
from spaceholes.io import attack_frame_positions, attack_frame_velocity, build_frame_context, load_events, load_tracking  # noqa: E402
from spaceholes.viz import plot_field, plot_players  # noqa: E402

MAIN = "prob_emp_A_tau0.5"


def field_at(tr, ctx, V, frame_rows):
    pos = attack_frame_positions(tr, ctx, frame_rows)
    off_v, def_v = attack_frame_velocity(tr, ctx, frame_rows)
    c = control_probabilistic(pos["off_xy"], off_v, pos["off_pe"], pos["off_det"], pos["def_xy"], def_v, pos["def_pe"], pos["def_det"])
    return pos, off_v, def_v, c * V[None, :]


def render_sequence(game, f0, offsets_s, title, fname, event_frames=None, tau=0.5):
    tr = load_tracking(game)
    ev = load_events(game)
    ctx = build_frame_context(tr, ev)
    V = np.load(CACHE_DIR / "value_maps_loo.npz")[str(game)]
    rowmap = tr.row_of_frame()
    pf = pd.read_parquet(OUT_DIR / f"frames_{game}.parquet")
    frames = [f0 + int(o * FPS) for o in offsets_s]
    rows = rowmap[np.clip(frames, 0, len(rowmap) - 1)]
    c0 = ctx.chance_idx[rowmap[f0]]
    keep = [i for i, r in enumerate(rows) if r >= 0 and ctx.chance_idx[r] == c0]
    if len(keep) < 3:
        return False
    offsets_s = [offsets_s[i] for i in keep]
    frames = [frames[i] for i in keep]
    rows = rows[keep]
    pos, off_v, def_v, H = field_at(tr, ctx, V, rows)
    n = len(frames)
    fig = plt.figure(figsize=(4.2 * n, 5.6))
    gs = fig.add_gridspec(2, n, height_ratios=[3.2, 1.2])
    vmax = max(0.9, float(H.max()))
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        im = plot_field(ax, H[i], vmax=vmax)
        handler = np.flatnonzero(pos["off_id"][i] == ctx.handler_id[rows[i]])
        plot_players(ax, pos["off_xy"][i], pos["def_xy"][i], pos["ball"][i], off_v[i], def_v[i], handler[0] if len(handler) else -1, occupancy_radius=5.0)
        ax.contour(GX, GY, to_grid(H[i]), levels=[tau], colors="white", linewidths=0.7, alpha=0.8)
        a = np.maximum(H[i] - tau, 0).sum()
        hole = float((np.maximum(H[i] - tau, 0) * (np.linalg.norm(GRID_Q[None] - pos["off_xy"][i][:, None], axis=-1).min(0) >= 5)).sum())
        ax.set_title(f"t = {offsets_s[i]:+.1f} s   A = {a:.0f}  (empty holes {hole:.0f})", fontsize=10)
    ax = fig.add_subplot(gs[1, :])
    cidx = ctx.chance_idx[rowmap[f0]]
    seg = pf[(pf.chance_idx == cidx)].sort_values("frame")
    t = (seg["frame"] - f0) / FPS
    ax.plot(t, seg[MAIN], color="#7b2cbf", lw=2, label="A(t): exploitable space (τ=0.5)")
    ax.plot(t, seg["vor_emp_A_tau0.5"], color="0.6", lw=1, ls="--", label="Voronoi baseline")
    for o in offsets_s:
        ax.axvline(o, color="0.7", lw=0.8)
    if event_frames:
        for f, lab in event_frames:
            ax.axvline((f - f0) / FPS, color="#e63946", lw=1)
            ax.text((f - f0) / FPS, ax.get_ylim()[1] * 0.95, lab, rotation=90, va="top", ha="right", fontsize=7, color="#e63946")
    ax.set_xlabel("seconds from event")
    ax.set_ylabel("A (ft²·pts)")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=12)
    fig.colorbar(im, ax=fig.axes[:n], fraction=0.015, pad=0.01, label="H = P(offense first) × value")
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def pick_examples(deltas):
    """One event per mechanism: clean tracking, strong but not extreme (85th-97th pct of gain within
    kind), peak within 2 s, distinct chances."""
    d = deltas.copy()
    d = d[(d[f"{MAIN}__n_valid"] >= 8) & (d["n_extrap__pre"] <= 1) & (d["n_extrap__post"] <= 1)
          & (d["off_pe_mean__pre"] < 2.0) & (d["def_pe_mean__pre"] < 2.0)]
    used = set()

    def strong(sub, col=f"{MAIN}__gain", lo=0.85, hi=0.97, ascending=False):
        sub = sub.dropna(subset=[col])
        q = sub[col].quantile([lo, hi]).values
        cand = sub[(sub[col] >= q[0]) & (sub[col] <= q[1])]
        if f"{MAIN}__t_peak" in cand:
            cand = cand[cand[f"{MAIN}__t_peak"].fillna(0) <= 2.0] if not ascending else cand
        cand = cand.sort_values(col, ascending=ascending)
        for _, r in cand.iterrows():
            key = (r["game_id"], r["chance_idx"])
            if key not in used:
                used.add(key)
                return r
        return cand.iloc[0]

    ex = {}
    ex["screen_opens_hole"] = strong(d[d.kind == "pick"])
    ex["offball_screen_opens_hole"] = strong(d[d.kind == "off_ball_screen"])
    ex["drive_relocates_hole"] = strong(d[d.kind == "drive"])
    ex["skip_pass_exploits_hole"] = strong(d[(d.kind == "pass") & (d["distance"].astype(float) > 25)], col=f"{MAIN}__delta")
    ex["closeout_repairs_hole"] = strong(d[d.kind == "closeout"], col=f"{MAIN}__delta", lo=0.03, hi=0.15, ascending=True)
    sh = d[(d.kind == "shot") & (d["outcome"] == True)]  # noqa: E712
    ex["made_shot_no_geometric_advantage"] = strong(sh, col=f"{MAIN}__pre", lo=0.03, hi=0.15, ascending=True)
    ob2 = d[(d.kind == "off_ball_screen") & (~d["ledToShot"].astype(bool))]
    ex["unrewarded_screen_big_advantage"] = strong(ob2)
    return ex


def main():
    deltas = pd.read_parquet(OUT_DIR / "event_deltas.parquet")
    ex = pick_examples(deltas)
    lines = []
    for name, e in ex.items():
        g, f0 = int(e["game_id"]), int(e["frame"])
        offs = [-3.0, -2.0, -1.0, -0.4, 0.0] if e["kind"] == "shot" else [-1.0, 0.0, 1.0, 2.0, 3.0]
        ok = render_sequence(g, f0, offs, f"{name.replace('_', ' ')} — game {g}, {e['kind']} {e['event_id']}", f"case_{name}.png")
        lines.append(f"{name}: game {g} {e['event_id']} frame {f0} pre={e[MAIN + '__pre']:.1f} post={e[MAIN + '__post']:.1f} "
                     f"gain={e[MAIN + '__gain']:.1f} half_life={e[MAIN + '__half_life']} rendered={ok}")
    (OUT_DIR / "case_studies.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
