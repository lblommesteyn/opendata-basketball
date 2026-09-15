"""Team control fields C_O(q,t) = P(offense reaches q before defense)."""

import numpy as np
from scipy.special import ndtr

from .config import GRID_Q
from .motion import tta_euclid, tta_kinematic, tta_sigma

T_GRID = np.arange(0.0, 6.001, 0.1, dtype=np.float32)  # arrival-time integration grid


def control_voronoi(off_p, def_p, q=GRID_Q):
    """Hard Euclidean Voronoi: 1 where the nearest player is offensive."""
    to = tta_euclid(off_p, q).min(axis=1)
    td = tta_euclid(def_p, q).min(axis=1)
    return (to < td).astype(np.float32)


def control_kinematic(off_p, off_v, def_p, def_v, q=GRID_Q, soft_scale=None, **motion_kw):
    """Velocity-aware deterministic control (hard), or logistic-soft if soft_scale (s) given."""
    to = tta_kinematic(off_p, off_v, q, **motion_kw).min(axis=1)
    td = tta_kinematic(def_p, def_v, q, **motion_kw).min(axis=1)
    if soft_scale is None:
        return (to < td).astype(np.float32)
    return (1.0 / (1.0 + np.exp((to - td) / soft_scale))).astype(np.float32)


def _min_survival(t_mean, t_sd, t_grid=T_GRID):
    """S(t) = P(min_i T_i > t) for independent normal arrival times. Returns (m, nt, n)."""
    z = (t_grid[None, None, :, None] - t_mean[:, :, None, :]) / t_sd[:, :, None, :]
    s_i = 1.0 - ndtr(z)  # (m,k,nt,n)
    return np.prod(s_i, axis=1)


def _clark_min(mu, sd):
    """Clark (1961) moment-matching approximation of min_i T_i for independent normals.

    mu, sd: (m,k,n). Returns (mu_min, sd_min) each (m,n).
    """
    m1, s1 = mu[:, 0, :], sd[:, 0, :]
    for i in range(1, mu.shape[1]):
        m2, s2 = mu[:, i, :], sd[:, i, :]
        a = np.sqrt(s1 * s1 + s2 * s2) + 1e-6
        alpha = (m1 - m2) / a
        phi = np.exp(-0.5 * alpha * alpha) / np.sqrt(2 * np.pi)
        Phi = ndtr(alpha)
        mmin = m1 * (1 - Phi) + m2 * Phi - a * phi
        m2nd = (m1 * m1 + s1 * s1) * (1 - Phi) + (m2 * m2 + s2 * s2) * Phi - (m1 + m2) * a * phi
        s1 = np.sqrt(np.maximum(m2nd - mmin * mmin, 1e-8))
        m1 = mmin
    return m1, s1


def control_probabilistic(off_p, off_v, off_pe, off_det, def_p, def_v, def_pe, def_det, q=GRID_Q, method="clark", t_grid=T_GRID, motion_kw=None, **sigma_kw):
    """Uncertainty-aware control: P(min_O T < min_D T) with T_i ~ N(mu_i, sigma_i^2).

    method='clark': moment-match each team's first-arrival time with Clark's recursion, then
    P = Phi((mu_D - mu_O) / sqrt(sd_O^2 + sd_D^2)).
    method='integral': exact-under-independence numerical integral of S_D(t) dF_O(t) on t_grid.
    """
    motion_kw = motion_kw or {}
    mu_o = tta_kinematic(off_p, off_v, q, **motion_kw)
    mu_d = tta_kinematic(def_p, def_v, q, **motion_kw)
    sd_o = tta_sigma(mu_o, off_pe, off_det, **sigma_kw)
    sd_d = tta_sigma(mu_d, def_pe, def_det, **sigma_kw)
    if method == "clark":
        mo, so = _clark_min(mu_o, sd_o)
        md, sd = _clark_min(mu_d, sd_d)
        return ndtr((md - mo) / np.sqrt(so * so + sd * sd)).astype(np.float32)
    s_o = _min_survival(mu_o, sd_o, t_grid)  # (m,nt,n)
    s_d = _min_survival(mu_d, sd_d, t_grid)
    f_o = -np.diff(s_o, axis=1)  # mass of offensive first-arrival in each t bin
    s_d_mid = 0.5 * (s_d[:, 1:, :] + s_d[:, :-1, :])
    p = (f_o * s_d_mid).sum(axis=1)
    # mass beyond the grid: both still absent; split by who is likelier to arrive first
    tail_o, tail_d = s_o[:, -1, :], s_d[:, -1, :]
    tail = tail_o * tail_d
    p = p + tail * (mu_d.min(axis=1) > mu_o.min(axis=1))
    return np.clip(p, 0.0, 1.0).astype(np.float32)


def nearest_defender_distance(off_p, def_p):
    """(m,5) distance from each offensive player to the closest defender."""
    d = np.linalg.norm(off_p[:, :, None, :] - def_p[:, None, :, :], axis=-1)
    return d.min(axis=2)
