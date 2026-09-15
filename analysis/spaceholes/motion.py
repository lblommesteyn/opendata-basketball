"""Per-player time-to-arrival models T_i(q).

All functions take player positions `p` (m, k, 2), velocities `v` (m, k, 2) and grid
points `q` (n, 2) in the attack frame and return arrival times of shape (m, k, n).
"""

import numpy as np

from .config import A_MAX, EXTRAP_INFLATE, KAPPA_T, REACTION_TIME, SIGMA_T0, V_MAX, V_REF


def tta_euclid(p, q, v_max=V_MAX):
    """Model 1: straight-line distance at top speed. Ordering equals Euclidean Voronoi."""
    d = np.linalg.norm(p[:, :, None, :] - q[None, None, :, :], axis=-1)
    return d / v_max


def tta_kinematic(p, v, q, v_max=V_MAX, a_max=A_MAX, reaction=REACTION_TIME):
    """Model 2: reaction-time drift, then 1-D bang-bang acceleration toward q.

    The player keeps velocity v for `reaction` seconds, then accelerates at a_max toward q
    (from the initial speed component along the line to q, which may be negative), capped at
    v_max. Perpendicular velocity is discarded.
    """
    p1 = p + v * reaction  # (m,k,2)
    diff = q[None, None, :, :] - p1[:, :, None, :]
    d = np.linalg.norm(diff, axis=-1)  # (m,k,n)
    u = diff / np.maximum(d, 1e-6)[..., None]
    s0 = np.einsum("mknd,mkd->mkn", u, v)  # signed speed along the line to q
    s0 = np.clip(s0, -v_max, v_max)
    # moving away: time to stop and extra distance to make up
    neg = s0 < 0
    t_stop = np.where(neg, -s0 / a_max, 0.0)
    d_eff = d + np.where(neg, s0 * s0 / (2 * a_max), 0.0)
    s0p = np.where(neg, 0.0, s0)
    # accelerate from s0p to v_max
    d_acc = (v_max * v_max - s0p * s0p) / (2 * a_max)
    t_reach_vmax = (v_max - s0p) / a_max
    t_short = (-s0p + np.sqrt(s0p * s0p + 2 * a_max * d_eff)) / a_max
    t_long = t_reach_vmax + (d_eff - d_acc) / v_max
    t = np.where(d_eff <= d_acc, t_short, t_long)
    return reaction + t_stop + t


def tta_sigma(t_mean, pred_error, detected, sigma0=SIGMA_T0, v_ref=V_REF, kappa=KAPPA_T, inflate=EXTRAP_INFLATE):
    """Model 3 arrival-time standard deviation (m,k,n).

    Combines irreducible decision noise, positional tracking error converted to time via a
    reference speed (inflated for extrapolated positions), and a term proportional to the
    arrival time itself (path / dynamics uncertainty grows with distance).
    """
    pe = np.asarray(pred_error, dtype=np.float32)
    pe = np.where(np.asarray(detected) == 1, pe, pe * inflate)
    s_pos = (pe / v_ref)[:, :, None]
    return np.sqrt(sigma0 ** 2 + s_pos ** 2 + (kappa * t_mean) ** 2)
