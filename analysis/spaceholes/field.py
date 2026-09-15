"""Exploitable-space field H = C_O * V and its scalar summaries."""

import numpy as np
from scipy import ndimage

from .config import CELL_AREA, FT_LINE_X, GRID_X, GRID_Y, GRID_Q, PAINT_HALF_WIDTH, RIM, THREE_CORNER_Y
from .value import is_three, rim_distance

NX, NY = len(GRID_X), len(GRID_Y)
_R = rim_distance(GRID_Q)
NEAR_RIM = _R <= 8.0
CORNER = (np.abs(GRID_Q[:, 1]) >= THREE_CORNER_Y - 3) & (GRID_Q[:, 0] <= RIM[0] + 8)
THREE = is_three(GRID_Q) & (GRID_Q[:, 0] <= 0) & (_R <= 30)
PAINT = (np.abs(GRID_Q[:, 1]) <= PAINT_HALF_WIDTH) & (GRID_Q[:, 0] <= FT_LINE_X)

TAUS = (0.3, 0.5, 0.7)


def to_grid(h):
    return h.reshape(NX, NY)


def summaries(h, taus=TAUS, off_xy=None, occupied_radius=5.0):
    """Scalar advantage measures for a batch of fields h (m, n_cells). Returns dict of (m,) arrays.

    If off_xy (m,5,2) is given, A_tau0.5 is also split into space within `occupied_radius` ft of an
    offensive player (A_occ: open players) and space nobody occupies (A_hole: true holes).
    """
    out = {}
    out["A_total"] = h.sum(1) * CELL_AREA
    if off_xy is not None:
        dmin = np.linalg.norm(GRID_Q[None, None, :, :] - off_xy[:, :, None, :], axis=-1).min(1)
        near = dmin < occupied_radius
        hp = np.maximum(h - 0.5, 0)
        out["A_occ"] = (hp * near).sum(1) * CELL_AREA
        out["A_hole"] = (hp * ~near).sum(1) * CELL_AREA
        out["A_hole_rim"] = (hp * ~near * NEAR_RIM[None, :]).sum(1) * CELL_AREA
    for tau in taus:
        out[f"A_tau{tau:.1f}"] = np.maximum(h - tau, 0).sum(1) * CELL_AREA
    out["A_max"] = h.max(1)
    out["A_rim"] = h[:, NEAR_RIM].sum(1) * CELL_AREA
    out["A_corner"] = h[:, CORNER].sum(1) * CELL_AREA
    out["A_three"] = h[:, THREE].sum(1) * CELL_AREA
    out["A_paint"] = h[:, PAINT].sum(1) * CELL_AREA
    # largest connected high-value region (mass above tau=0.5)
    cc = np.zeros(len(h), dtype=np.float32)
    ncomp = np.zeros(len(h), dtype=np.int16)
    for i in range(len(h)):
        g = to_grid(np.maximum(h[i] - 0.5, 0))
        lab, n = ndimage.label(g > 0)
        ncomp[i] = n
        if n:
            cc[i] = ndimage.sum(g, lab, index=np.arange(1, n + 1)).max() * CELL_AREA
    out["A_cc"] = cc
    out["n_holes"] = ncomp
    return out
