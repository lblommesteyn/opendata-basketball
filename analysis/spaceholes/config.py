from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # .../analysis
REPO = ROOT.parent
# In the SkillCorner fork the data lives at the repository root (data/matches); when this package
# is used from another project, point it at a clone via data/external/opendata-basketball/data.
DATA_DIR = REPO / "data" if (REPO / "data" / "matches").exists() else ROOT / "data" / "external" / "opendata-basketball" / "data"
MATCH_DIR = DATA_DIR / "matches"
CACHE_DIR = ROOT / "cache"
OUT_DIR = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
PAPER_DIR = ROOT / "paper"

for d in (CACHE_DIR, OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

GAME_IDS = [114086, 114099, 114169, 114234, 114243, 178442, 179612, 184439, 188630, 191313]

FPS = 25
DT = 1.0 / FPS

# FIBA court in feet (28 m x 15 m), origin at centre court. Attack frame: offensive
# hoop at negative x (matches the SkillCorner event-table convention).
COURT_LENGTH = 28.0 / 0.3048  # 91.86 ft
COURT_WIDTH = 15.0 / 0.3048  # 49.21 ft
HALF_LENGTH = COURT_LENGTH / 2
HALF_WIDTH = COURT_WIDTH / 2
RIM_X = -(HALF_LENGTH - 1.575 / 0.3048)  # -40.76 ft (rim centre is 1.575 m from baseline)
RIM = np.array([RIM_X, 0.0])
THREE_RADIUS = 6.75 / 0.3048  # 22.15 ft
THREE_CORNER_Y = HALF_WIDTH - 0.90 / 0.3048  # 21.65 ft: straight corner segment
RA_RADIUS = 1.25 / 0.3048  # restricted area 4.1 ft
PAINT_HALF_WIDTH = 2.45 / 0.3048  # 8.04 ft
PAINT_DEPTH = 5.8 / 0.3048  # 19.03 ft from baseline
FT_LINE_X = -(HALF_LENGTH - PAINT_DEPTH)

# Analysis grid over the offensive half court (attack frame), 1 ft cells.
GRID_RES = 1.0
GRID_X = np.arange(-HALF_LENGTH + 0.5, 0.0, GRID_RES)  # baseline .. half court
GRID_Y = np.arange(-HALF_WIDTH + 0.5, HALF_WIDTH, GRID_RES)
GX, GY = np.meshgrid(GRID_X, GRID_Y, indexing="ij")
GRID_Q = np.stack([GX.ravel(), GY.ravel()], axis=1)  # (n_cells, 2)
CELL_AREA = GRID_RES * GRID_RES

# Movement model constants, fitted from the tracking data (see outputs/holes/movement_fit.json).
V_MAX = 20.0  # ft/s (99.9th percentile of smoothed speed)
A_MAX = 20.0  # ft/s^2 (99th percentile of smoothed acceleration)
REACTION_TIME = 0.2  # s: time before a player can change their motion
SIGMA_T0 = 0.15  # s: irreducible arrival-time noise (decision / reaction variability)
V_REF = 10.0  # ft/s: converts positional error (ft) to arrival-time error (s)
KAPPA_T = 0.10  # arrival-time noise grows by this fraction of the arrival time itself
EXTRAP_INFLATE = 1.5  # extra inflation of positional error for extrapolated positions

# Analysis sampling
SAMPLE_STRIDE = 5  # evaluate fields every 5th frame (5 Hz) for game-wide passes
