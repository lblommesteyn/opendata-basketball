"""Court value functions V(q): the value to the offense of controlling location q."""

import numpy as np

from .config import GRID_Q, HALF_WIDTH, RIM, THREE_CORNER_Y, THREE_RADIUS


def is_three(q):
    """True where a shot from q is worth three (FIBA arc with straight corner segments)."""
    dx = q[:, 0] - RIM[0]
    dy = q[:, 1]
    r = np.hypot(dx, dy)
    corner = np.abs(dy) >= THREE_CORNER_Y
    # corner segment: straight line at |y| = THREE_CORNER_Y from baseline until it meets the arc
    return np.where(corner, True, r >= THREE_RADIUS)


def rim_distance(q):
    return np.hypot(q[:, 0] - RIM[0], q[:, 1] - RIM[1])


def value_geometric(q=GRID_Q):
    """Parametric expected points of an *open* shot from q (interpretable baseline).

    Rim 1.35 pts, decays to 0.85 at 15 ft, three-point zone 1.05 at the line decaying to
    0.35 at 35 ft, zero beyond half court / far backcourt.
    """
    r = rim_distance(q)
    three = is_three(q)
    two = 1.35 - 0.5 * np.clip(r / 15.0, 0, 1)
    thr = 1.05 - 0.70 * np.clip((r - THREE_RADIUS) / 13.0, 0, 1)
    v = np.where(three, thr, two)
    v = np.where(r > 40.0, 0.05, v)
    v = np.where(q[:, 0] > 0, 0.0, v)  # backcourt
    return v.astype(np.float32)


def value_uniform(q=GRID_Q):
    """1 on the offensive half court: area-only control (pure Voronoi-area baseline)."""
    return (q[:, 0] <= 0).astype(np.float32)


def fit_empirical_value(shots, q=GRID_Q, bandwidth=4.0, prior_weight=6.0, open_only=True, mirror=True):
    """Kernel-smoothed expected points of open/light shots by location, shrunk to value_geometric.

    shots: DataFrame with `location`, `three`, `outcome`, `contestLevel`, `fouled`.
    Points on a fouled miss are ignored (0), fouled makes count the field goal only.
    Returns V (n_cells,) and the effective sample count per cell.
    """
    s = shots.dropna(subset=["location"]).copy()
    if open_only:
        s = s[s["contestLevel"].isin(["open", "light"])]
    loc = np.array([list(l) for l in s["location"]], dtype=float)
    pts = (s["outcome"].astype(bool) * np.where(s["three"].astype(bool), 3, 2)).astype(float).values
    if mirror:
        loc = np.vstack([loc, loc * np.array([1, -1])])
        pts = np.concatenate([pts, pts])
    d2 = ((q[:, None, :] - loc[None, :, :]) ** 2).sum(-1)
    w = np.exp(-0.5 * d2 / bandwidth ** 2)
    # a shot's value transfers within its 2/3 region only
    same_zone = is_three(q)[:, None] == is_three(loc)[None, :]
    w = w * same_zone
    n_eff = w.sum(1)
    prior = value_geometric(q)
    v = (w @ pts + prior_weight * prior) / (n_eff + prior_weight)
    v = np.where(q[:, 0] > 0, 0.0, v)
    return v.astype(np.float32), n_eff


def ball_reach_weight(q, ball_xy, d0=8.0, scale=25.0):
    """Discount for locations far from the ball: exp(-(d - d0)+ / scale). (m, n)."""
    d = np.linalg.norm(q[None, :, :] - ball_xy[:, None, :], axis=-1)
    return np.exp(-np.maximum(d - d0, 0) / scale).astype(np.float32)
