"""Stage 10 (prototype): the defense's weakest cut.

Grid graph over the offensive half court, edge capacity = min(P(offense first)) of the two cells,
source = cells around the ball, sink = cells within 5 ft of the rim. Max flow is 'lane width'; the
min cut is the wall of defended space between the ball and the rim, drawn as the boundary between
the source side and the sink side of the residual graph.

Usage: python scripts/holes/10_mincut.py [game] [frame] [t_before] [t_after] [name] [title]
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import ndimage

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spaceholes.config import CACHE_DIR, FIG_DIR, FPS, GRID_Q, GRID_X, GRID_Y, RIM  # noqa: E402
from spaceholes.control import control_probabilistic  # noqa: E402
from spaceholes.field import NX, NY, to_grid  # noqa: E402
from spaceholes.io import attack_frame_positions, attack_frame_velocity, build_frame_context, load_events, load_tracking  # noqa: E402
from spaceholes.viz import plot_field  # noqa: E402

BG = "#0b0d12"
OFF, DEF, BALL, ACC, CUT = "#5aa9ff", "#ff5d52", "#f5b041", "#c084fc", "#00e5a0"
RIM_R = 5.0
SRC_R = 2.5
STEP = 2  # grid stride (2 ft cells) keeps the flow solve fast


def widest_path(cgrid, ball_xy):
    """Maximin path from the ball to the rim zone on the STEP grid.

    Returns (bottleneck value b*, side mask = cells reachable from the ball with control > b*,
    path as list of (x, y), bottleneck point (x, y), xs, ys)."""
    import heapq
    xs, ys = GRID_X[::STEP], GRID_Y[::STEP]
    c = cgrid[::STEP, ::STEP].copy()
    nx_, ny_ = c.shape
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    # the on-ball contest is not the lane: the handler already has the ball where he stands
    c[np.hypot(X - ball_xy[0], Y - ball_xy[1]) <= 3.0] = 1.0
    if np.hypot(ball_xy[0] - RIM[0], ball_xy[1] - RIM[1]) <= RIM_R + 3.0:
        return np.nan, np.zeros_like(c, dtype=bool), [], None, xs, ys
    bi, bj = int(np.argmin(np.abs(xs - ball_xy[0]))), int(np.argmin(np.abs(ys - ball_xy[1])))
    snk = np.hypot(X - RIM[0], Y - RIM[1]) <= RIM_R
    best = np.full(c.shape, -1.0)
    prev = {}
    best[bi, bj] = c[bi, bj]
    heap = [(-c[bi, bj], bi, bj)]
    target = None
    while heap:
        nb, i, j = heapq.heappop(heap)
        nb = -nb
        if nb < best[i, j]:
            continue
        if snk[i, j]:
            target = (i, j)
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx_ and 0 <= b < ny_:
                v = min(nb, c[a, b])
                if v > best[a, b]:
                    best[a, b] = v
                    prev[(a, b)] = (i, j)
                    heapq.heappush(heap, (-v, a, b))
    if target is None:
        return 0.0, np.zeros_like(c, dtype=bool), [], None, xs, ys
    bstar = float(best[target])
    path, node = [], target
    while node is not None:
        path.append((xs[node[0]], ys[node[1]]))
        node = prev.get(node)
    path = path[::-1]
    # bottleneck point: first cell along the path whose control equals b*
    bn = None
    for (x, y) in path:
        i, j = int(np.argmin(np.abs(xs - x))), int(np.argmin(np.abs(ys - y)))
        if c[i, j] <= bstar + 1e-6:
            bn = (x, y)
            break
    # the wall: the ball's connected component when everything at or below b* is defended
    lab, _ = ndimage.label(c > bstar + 1e-6)
    side = lab == lab[bi, bj] if lab[bi, bj] > 0 else np.zeros_like(c, dtype=bool)
    return bstar, side, path, bn, xs, ys


def cut_boundary(side, xs, ys):
    """Segments (as x,y pairs) between source-side and sink-side cells."""
    segs = []
    h = STEP / 2
    for i in range(side.shape[0]):
        for j in range(side.shape[1]):
            if not side[i, j]:
                continue
            x, y = xs[i], ys[j]
            if i + 1 < side.shape[0] and not side[i + 1, j]:
                segs.append(((x + h, y - h), (x + h, y + h)))
            if i - 1 >= 0 and not side[i - 1, j]:
                segs.append(((x - h, y - h), (x - h, y + h)))
            if j + 1 < side.shape[1] and not side[i, j + 1]:
                segs.append(((x - h, y + h), (x + h, y + h)))
            if j - 1 >= 0 and not side[i, j - 1]:
                segs.append(((x - h, y - h), (x + h, y - h)))
    return segs


def render(game=188630, f0=24725, t_before=2.0, t_after=3.0, name="lane_drive", fps_out=8, title="The widest path to the rim and the wall that seals it", event_label="the drive"):
    tr, ev = load_tracking(game), load_events(game)
    ctx = build_frame_context(tr, ev)
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
    t = (frames - f0) / FPS
    flows, sides, cuts, paths, bns = [], [], [], [], []
    for i in range(len(frames)):
        cg = to_grid(C[i])
        val, side, path, bn, xs, ys = widest_path(cg, pos["ball"][i][:2])
        flows.append(val)
        sides.append(side)
        cuts.append(cut_boundary(side, xs, ys))
        paths.append(path)
        bns.append(bn)
    flows = np.array(flows)
    fmax = np.nanmax(flows)

    fig = plt.figure(figsize=(9, 8.2), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.6, 1.5], hspace=0.12, left=0.04, right=0.96, top=0.93, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    axl = fig.add_subplot(gs[1])

    def draw(i):
        ax.clear()
        axl.clear()
        ax.set_facecolor(BG)
        plot_field(ax, C[i], vmax=1.0, cmap="magma")
        # source side tint
        side_full = np.kron(sides[i], np.ones((STEP, STEP), dtype=bool))[:NX, :NY]
        ax.contourf(GRID_X, GRID_Y, side_full.T.astype(float), levels=[0.5, 1.5], colors=[ACC], alpha=0.18)
        for (x0, y0), (x1, y1) in cuts[i]:
            ax.plot([x0, x1], [y0, y1], color=CUT, lw=2.6, solid_capstyle="round", zorder=7)
        if paths[i]:
            px, py = zip(*paths[i])
            ax.plot(px, py, color="white", lw=2.2, alpha=0.9, zorder=7, ls=(0, (4, 3)))
        if bns[i] is not None:
            ax.scatter([bns[i][0]], [bns[i][1]], s=260, facecolors="none", ec=CUT, lw=2.5, zorder=9)
        ax.add_patch(plt.Circle((RIM[0], RIM[1]), RIM_R, fill=False, ec="white", lw=1, ls="--", alpha=0.6))
        ax.scatter(pos["def_xy"][i][:, 0], pos["def_xy"][i][:, 1], s=140, c=DEF, ec="white", lw=1.2, zorder=8)
        ax.scatter(pos["off_xy"][i][:, 0], pos["off_xy"][i][:, 1], s=140, c=OFF, ec="white", lw=1.2, zorder=8)
        ax.quiver(pos["off_xy"][i][:, 0], pos["off_xy"][i][:, 1], ov[i][:, 0], ov[i][:, 1], color=OFF, scale=90, width=0.004, zorder=6)
        ax.quiver(pos["def_xy"][i][:, 0], pos["def_xy"][i][:, 1], dv[i][:, 0], dv[i][:, 1], color=DEF, scale=90, width=0.004, zorder=6)
        b = pos["ball"][i]
        ax.scatter([b[0]], [b[1]], s=70, c=BALL, ec="black", zorder=9)
        ax.text(0.02, 0.97, f"t = {t[i]:+.1f} s", transform=ax.transAxes, color="white", fontsize=15, fontweight="bold", va="top", bbox=dict(facecolor=BG, alpha=0.75, edgecolor="none", pad=4), zorder=10)
        ax.text(0.98, 0.97, "at the rim" if np.isnan(flows[i]) else f"lane openness = {flows[i]:.2f}", transform=ax.transAxes, color=CUT, fontsize=17, fontweight="bold", va="top", ha="right", bbox=dict(facecolor=BG, alpha=0.75, edgecolor="none", pad=4), zorder=10)
        ax.set_title(title, color="white", fontsize=12.5, pad=8)
        axl.set_facecolor(BG)
        axl.plot(t, flows, color=CUT, lw=1.2, alpha=0.25)
        axl.plot(t[: i + 1], flows[: i + 1], color=CUT, lw=2.6)
        if not np.isnan(flows[i]):
            axl.scatter([t[i]], [flows[i]], s=60, c=CUT, zorder=5)
        axl.axvline(0, color="white", lw=0.7, alpha=0.5)
        axl.text(0.05, 0.95, f"{event_label} starts", color="white", fontsize=8, va="top", alpha=0.85)
        axl.set_xlim(t[0], t[-1])
        axl.set_ylim(0, 1.0)
        axl.set_xlabel(f"seconds from {event_label}", color="white", fontsize=10)
        axl.set_ylabel("P(offense first) at the bottleneck", color="white", fontsize=10)
        axl.tick_params(colors="white", labelsize=9)
        for sp in axl.spines.values():
            sp.set_color("#3a4150")
        axl.text(0.995, 0.06, "dashed = widest path to the rim, ring = its bottleneck  |  green = the wall: where the defense seals the ball's region", transform=axl.transAxes, color="white", fontsize=8, alpha=0.7, ha="right")
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    out = FIG_DIR / f"social_{name}.gif"
    anim.save(out, writer=PillowWriter(fps=fps_out), savefig_kwargs=dict(facecolor=BG))
    for k, i in {"pre": 0, "peak": int(np.nanargmax(flows)), "min": int(np.nanargmin(flows))}.items():
        draw(i)
        fig.savefig(FIG_DIR / f"social_{name}_{k}.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(out, len(frames), "frames", f"{out.stat().st_size / 1e6:.1f} MB", "flow range", np.nanmin(flows).round(1), np.nanmax(flows).round(1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a:
        render(int(a[0]), int(a[1]), float(a[2]), float(a[3]), a[4], title=a[5] if len(a) > 5 else "", event_label=a[6] if len(a) > 6 else "the action")
    else:
        render()
