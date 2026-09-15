"""Exploitable-space / defensive-hole model for SkillCorner basketball tracking.

Modules
-------
config   court geometry, grid, paths, movement constants
io       loading + caching of tracking and events, orientation to attack frame
motion   per-player time-to-arrival models (Euclidean, velocity-aware, probabilistic)
control  team control fields C_O(q,t) from the motion models
value    court value functions V(q)
field    H = C * V and scalar advantage summaries A(t)
"""
