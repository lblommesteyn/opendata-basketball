"""Stage 9: social media assets — an animated GIF of the field through a possession and a hero still.

Usage: python scripts/holes/09_social_media.py [game] [frame] [t_before] [t_after] [name]
Defaults to the 'drive relocates hole' case study.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spaceholes.config import CACHE_DIR, FIG_DIR, FPS, GRID_Q, GX, GY  # noqa: E402
from spaceholes.control import control_probabilistic  # noqa: E402
from spaceholes.events import collect_events  # noqa: E402
from spaceholes.field import to_grid  # noqa: E402
from spaceholes.io import attack_frame_positions, attack_frame_velocity, build_frame_context, load_events, load_tracking  # noqa: E402
from spaceholes.viz import plot_field  # noqa: E402

BG = "#0b0d12"
OFF, DEF, BALL, ACC = "#5aa9ff", "#ff5d52", "#f5b041", "#c084fc"


def render(game=188630, f0=24725, t_before=2.0, t_after=4.0, name="drive", fps_out=10, title="A drive collapses the defense and the hole moves to the weak side", event_label="the drive"):
    tr, ev = load_tracking(game), load_events(game)
    ctx = build_frame_context(tr, ev)
    V = np.load(CACHE_DIR / "value_maps_loo.npz")[str(game)]
    rowmap = tr.row_of_frame()
    step = FPS // fps_out
    frames = np.arange(f0 - int(t_before * FPS), f0 + int(t_after * FPS) + 1, step)
    rows = rowmap[frames]
    c0 = ctx.chance_idx[rowmap[f0]]
    keep = (rows >= 0) & (ctx.chance_idx[np.clip(rows, 0, None)] == c0)
    frames, rows = frames[keep], rows[keep]
    pos = attack_frame_positions(tr, ctx, rows)
    ov, dv = attack_frame_velocity(tr, ctx, rows)
    C = control_probabilistic(pos["off_xy"], ov, pos["off_pe"], pos["off_det"], pos["def_xy"], dv, pos["def_pe"], pos["def_det"])
    H = C * V[None, :]
    A = np.maximum(H - 0.5, 0).sum(1)
    dmin = np.linalg.norm(GRID_Q[None, None] - pos["off_xy"][:, :, None], axis=-1).min(1)
    A_hole = (np.maximum(H - 0.5, 0) * (dmin >= 5)).sum(1)
    t = (frames - f0) / FPS
    events = collect_events(ev, game)
    ch = ev["chances"].reset_index()
    cmap = dict(zip(ch["id"], ch["index"]))
    events["chance_idx"] = events["chanceId"].map(cmap).fillna(-1).astype(int)
    evs = events[(events.chance_idx == c0) & (events.frame >= frames[0]) & (events.frame <= frames[-1]) & (~events.kind.isin(["pass"]))]

    fig = plt.figure(figsize=(9, 8.2), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.6, 1.5], hspace=0.12, left=0.04, right=0.96, top=0.93, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    axl = fig.add_subplot(gs[1])
    vmax = max(1.0, float(np.percentile(H, 99.5)))

    def draw(i):
        ax.clear()
        axl.clear()
        ax.set_facecolor(BG)
        plot_field(ax, H[i], vmax=vmax)
        ax.contour(GX, GY, to_grid(H[i]), levels=[0.5], colors="white", linewidths=0.8, alpha=0.8)
        for x, y in pos["off_xy"][i]:
            ax.add_patch(plt.Circle((x, y), 5, fill=False, ec=OFF, lw=0.6, ls=":", alpha=0.6))
        ax.scatter(pos["def_xy"][i][:, 0], pos["def_xy"][i][:, 1], s=140, c=DEF, ec="white", lw=1.2, zorder=5)
        ax.scatter(pos["off_xy"][i][:, 0], pos["off_xy"][i][:, 1], s=140, c=OFF, ec="white", lw=1.2, zorder=5)
        ax.quiver(pos["off_xy"][i][:, 0], pos["off_xy"][i][:, 1], ov[i][:, 0], ov[i][:, 1], color=OFF, scale=90, width=0.004, zorder=4)
        ax.quiver(pos["def_xy"][i][:, 0], pos["def_xy"][i][:, 1], dv[i][:, 0], dv[i][:, 1], color=DEF, scale=90, width=0.004, zorder=4)
        b = pos["ball"][i]
        ax.scatter([b[0]], [b[1]], s=55, c=BALL, ec="black", zorder=6)
        h = np.flatnonzero(pos["off_id"][i] == ctx.handler_id[rows[i]])
        if len(h):
            ax.scatter([pos["off_xy"][i][h[0], 0]], [pos["off_xy"][i][h[0], 1]], s=320, facecolors="none", ec="#ffe66d", lw=2.2, zorder=6)
        ax.text(0.02, 0.97, f"t = {t[i]:+.1f} s", transform=ax.transAxes, color="white", fontsize=15, fontweight="bold", va="top", bbox=dict(facecolor=BG, alpha=0.75, edgecolor="none", pad=4), zorder=8)
        ax.text(0.98, 0.97, f"A = {A[i]:.0f}", transform=ax.transAxes, color=ACC, fontsize=17, fontweight="bold", va="top", ha="right", bbox=dict(facecolor=BG, alpha=0.75, edgecolor="none", pad=4), zorder=8)
        ax.text(0.98, 0.905, f"empty holes {A_hole[i]:.0f}", transform=ax.transAxes, color="white", fontsize=11, va="top", ha="right", bbox=dict(facecolor=BG, alpha=0.75, edgecolor="none", pad=3), zorder=8)
        ax.set_title(title, color="white", fontsize=13, pad=8)
        # A(t) trace
        axl.set_facecolor(BG)
        axl.plot(t, A, color=ACC, lw=1.2, alpha=0.25)
        axl.plot(t[: i + 1], A[: i + 1], color=ACC, lw=2.6)
        axl.fill_between(t[: i + 1], 0, A_hole[: i + 1], color=ACC, alpha=0.25, lw=0)
        axl.scatter([t[i]], [A[i]], s=60, c=ACC, zorder=5)
        for e in evs.itertuples():
            te = (e.frame - f0) / FPS
            axl.axvline(te, color="white", lw=0.7, alpha=0.5)
            axl.text(te + 0.05, A.max() * 0.95, e.kind.replace("_", " "), color="white", fontsize=8, va="top", alpha=0.85)
        axl.set_xlim(t[0], t[-1])
        axl.set_ylim(0, A.max() * 1.08)
        axl.set_xlabel(f"seconds from {event_label}", color="white", fontsize=10)
        axl.set_ylabel("exploitable space A", color="white", fontsize=10)
        axl.tick_params(colors="white", labelsize=9)
        for sp in axl.spines.values():
            sp.set_color("#3a4150")
        axl.text(0.995, 0.06, "bright = space the offense reaches first, worth shooting from  |  shaded = empty holes", transform=axl.transAxes,
                 color="white", fontsize=8, alpha=0.7, ha="right")
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    out = FIG_DIR / f"social_{name}.gif"
    anim.save(out, writer=PillowWriter(fps=fps_out), savefig_kwargs=dict(facecolor=BG))
    # hero still: the peak frame
    draw(int(np.argmax(A)))
    fig.savefig(FIG_DIR / f"social_{name}_still.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(out, len(frames), "frames", f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a:
        render(int(a[0]), int(a[1]), float(a[2]), float(a[3]), a[4], title=a[5] if len(a) > 5 else "", event_label=a[6] if len(a) > 6 else "the action")
    else:
        render()
