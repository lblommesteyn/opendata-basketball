# Holes in the Defense — exploitable-space analysis on the SkillCorner ACB release

This fork adds `analysis/`: a reproducible pipeline that builds an uncertainty-aware
space-control field from the 10 public ACB games in this repository and tests whether basketball
possessions can be described as the creation, movement and closure of valuable holes in the defense.

* `paper/spatial_holes.md` / `.pdf` — the write-up (also published at https://www.lukeblom.fyi/writing/holes-in-the-defense.html)
* `spaceholes/` — library: I/O and orientation, kinematic time-to-arrival, Clark closed-form control
  probability, court value maps, field summaries, event alignment, plotting (uses `mplbasketball`)
* `scripts/` — stages 01–08 (fields → events → validation → robustness → case studies → figures →
  action values → hole lifetimes) and the paper build; see `scripts/README.md`
* `results/` — summary tables produced by the stages
* `replay/` — an interactive HTML replay of eight possessions (open `index.html`; `data.js` carries
  the fields)

Run from the repository root with Python 3.10+, `numpy pandas scipy scikit-learn matplotlib pyarrow mplbasketball`:

```
python analysis/scripts/01_compute_fields.py      # ~4 min per game
python analysis/scripts/02_event_analysis.py
python analysis/scripts/03_validation.py
python analysis/scripts/04_robustness.py
python analysis/scripts/07_action_values.py
python analysis/scripts/05_case_studies.py
python analysis/scripts/06_figures.py
python analysis/scripts/08_hole_lifetimes.py
python analysis/scripts/build_spatial_holes.py   # needs pandoc + LaTeX
```

Data and its licence are SkillCorner's (see the top-level README); the analysis code is MIT.
