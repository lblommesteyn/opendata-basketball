# Exploitable-space ("holes") pipeline

Builds and validates an uncertainty-aware space-control field from SkillCorner ACB tracking
(the `data/` directory of this repository). Library code lives in `analysis/spaceholes/`; these
scripts run the stages in order and write to `analysis/results/` (tables), `analysis/paper/figures/`
(figures) and `analysis/cache/` (per-game caches, git-ignored).

```powershell
python .\analysis\scripts\01_compute_fields.py    # ~4 min / game: control fields, H summaries, baselines
python .\analysis\scripts\02_event_analysis.py    # delta-A around events, curves, half-lives, null control
python .\analysis\scripts\03_validation.py        # game-held-out prediction of next-3s outcomes
python .\analysis\scripts\04_robustness.py        # sensitivity to tracking error, model parameters, grid, tau
python .\analysis\scripts\05_case_studies.py      # possession-level examples with H(x,y,t) panels
python .\analysis\scripts\06_figures.py           # summary figures
python .\analysis\scripts\07_action_values.py     # action values in the common currency + label checks
python .\analysis\scripts\build_spatial_holes.py  # paper PDF
```

Frame convention: every field is computed in the *attack frame* (offensive hoop at negative x,
offensive left = positive y), obtained from the broadcast frame with the possession's `leftHoop`
flag (`leftHoop=False` ⇒ rotate 180°; verified against event `location`s, max error 0.2 ft).
Fields are evaluated at 5 Hz on live, half-court frames (game clock running, chance has reached
the frontcourt) over a 1 ft grid of the offensive half.
