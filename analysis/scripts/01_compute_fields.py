"""Stage 1: compute control fields, H summaries and baseline features for every game.

Writes outputs/holes/frames_{game}.parquet and data/processed/holes/fields_{game}.npz.
The empirical value map for each game is fitted on the *other* nine games (leave-one-game-out).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spaceholes.config import CACHE_DIR, GAME_IDS, OUT_DIR  # noqa: E402
from spaceholes.io import load_events  # noqa: E402
from spaceholes.pipeline import compute_game  # noqa: E402
from spaceholes.value import fit_empirical_value, value_geometric, value_uniform  # noqa: E402


def main(games=None):
    games = games or GAME_IDS
    shots = {g: load_events(g)["shots"] for g in GAME_IDS}
    value_store = {}
    for g in games:
        t0 = time.time()
        others = pd.concat([shots[o] for o in GAME_IDS if o != g], ignore_index=True)
        v_emp, n_eff = fit_empirical_value(others, bandwidth=5.0, prior_weight=4.0)
        maps = {"geom": value_geometric(), "unif": value_uniform(), "emp": v_emp}
        value_store[g] = v_emp
        df, fields = compute_game(g, maps)
        df.to_parquet(OUT_DIR / f"frames_{g}.parquet", index=False)
        np.savez_compressed(CACHE_DIR / f"fields_{g}.npz", rows=df["row"].values, **fields)
        print(f"{g}: {len(df)} frames, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(CACHE_DIR / "value_maps_loo.npz", **{str(k): v for k, v in value_store.items()})


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or None)
