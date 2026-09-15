"""Court drawing and field visualisation helpers (attack frame, offense attacks negative x)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle

from .config import FT_LINE_X, GRID_X, GRID_Y, HALF_LENGTH, HALF_WIDTH, PAINT_HALF_WIDTH, RA_RADIUS, RIM, THREE_CORNER_Y, THREE_RADIUS
from .field import to_grid


def draw_half_court(ax, color="0.3", lw=1.2):
    """FIBA half court in the attack frame (hoop at negative x), drawn with mplbasketball when
    available (its centre-origin FIBA court coincides with our frame to within 0.05 ft)."""
    try:
        from mplbasketball import Court
        Court(court_type="fiba", origin="center", units="ft").draw(ax=ax, orientation="hl", line_color=color, line_width=lw * 0.12, pad=0)
    except Exception:
        ax.add_patch(Rectangle((-HALF_LENGTH, -HALF_WIDTH), HALF_LENGTH, 2 * HALF_WIDTH, fill=False, ec=color, lw=lw))
        ax.add_patch(Rectangle((-HALF_LENGTH, -PAINT_HALF_WIDTH), FT_LINE_X + HALF_LENGTH, 2 * PAINT_HALF_WIDTH, fill=False, ec=color, lw=lw))
        ax.add_patch(Circle((RIM[0], 0), 0.75, fill=False, ec=color, lw=lw))
        ax.add_patch(Arc((RIM[0], 0), 2 * RA_RADIUS, 2 * RA_RADIUS, theta1=-90, theta2=90, ec=color, lw=lw))
        ax.add_patch(Arc((FT_LINE_X, 0), 12, 12, theta1=-90, theta2=90, ec=color, lw=lw))
        ang = np.degrees(np.arcsin(THREE_CORNER_Y / THREE_RADIUS))
        ax.add_patch(Arc((RIM[0], 0), 2 * THREE_RADIUS, 2 * THREE_RADIUS, theta1=-ang, theta2=ang, ec=color, lw=lw))
        x_meet = RIM[0] + np.sqrt(THREE_RADIUS ** 2 - THREE_CORNER_Y ** 2)
        for s in (-1, 1):
            ax.plot([-HALF_LENGTH, x_meet], [s * THREE_CORNER_Y, s * THREE_CORNER_Y], color=color, lw=lw)
    ax.set_xlim(-HALF_LENGTH - 1, 1)
    ax.set_ylim(-HALF_WIDTH - 1, HALF_WIDTH + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("on")
    for sp in ax.spines.values():
        sp.set_visible(True)


def plot_field(ax, h, vmax=None, cmap="magma", alpha=0.9):
    g = to_grid(h)
    ext = [GRID_X[0] - 0.5, GRID_X[-1] + 0.5, GRID_Y[0] - 0.5, GRID_Y[-1] + 0.5]
    im = ax.imshow(g.T, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=vmax, alpha=alpha, interpolation="bilinear")
    draw_half_court(ax, color="white" if cmap == "magma" else "0.2")
    return im


def plot_players(ax, off_xy, def_xy, ball=None, off_v=None, def_v=None, handler_idx=None, labels=None, occupancy_radius=None):
    if occupancy_radius:
        for x, y in off_xy:
            ax.add_patch(Circle((x, y), occupancy_radius, fill=False, ec="#1f77b4", lw=0.6, ls=":", alpha=0.7, zorder=4))
    ax.scatter(off_xy[:, 0], off_xy[:, 1], s=90, c="#1f77b4", ec="white", zorder=5, label="offense")
    ax.scatter(def_xy[:, 0], def_xy[:, 1], s=90, c="#d62728", ec="white", zorder=5, label="defense")
    if off_v is not None:
        ax.quiver(off_xy[:, 0], off_xy[:, 1], off_v[:, 0], off_v[:, 1], color="#1f77b4", scale=80, width=0.004, zorder=4)
    if def_v is not None:
        ax.quiver(def_xy[:, 0], def_xy[:, 1], def_v[:, 0], def_v[:, 1], color="#d62728", scale=80, width=0.004, zorder=4)
    if ball is not None:
        ax.scatter([ball[0]], [ball[1]], s=40, c="orange", ec="black", zorder=6)
    if handler_idx is not None and handler_idx >= 0:
        ax.scatter([off_xy[handler_idx, 0]], [off_xy[handler_idx, 1]], s=200, facecolors="none", ec="yellow", lw=2, zorder=6)
    if labels is not None:
        for (x, y), t in zip(off_xy, labels):
            ax.annotate(str(t), (x, y), fontsize=6, color="white", ha="center", va="center", zorder=7)
