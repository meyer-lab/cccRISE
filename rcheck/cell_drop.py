"""Leaving out one cell, with no randomness anywhere: modules of upstream R
(`cell_drop.R`) and the Python port on each sample's full CCIM, all cells
against each single cell left out. Prints whether R and Python agree exactly,
each run's module agreement with the all-cell modules
(`singlecell_joint.module_stability`), and why the tree cut is this sensitive:
the soft power, the spread of the merge heights and their typical gap, against
the largest change in the LR similarity.

    uv run python rcheck/cell_drop.py rcheck/out/drop rcheck/out/drop/r 0 37 74
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from cccrise.validation.singlecell_joint import module_stability

from cccrise.comparators import _wgcna, scriabin
from cccrise.comparators.inputs import entity_expression


def _similarity(expr, lr: pd.DataFrame, names: pd.Index) -> pd.DataFrame:
    """The full-CCIM LR similarity (``InteractionPrograms`` without
    subsampling), inputs built as `compare.py` does."""
    lig = entity_expression(expr, names, lr["ligand"], "min")
    rec = entity_expression(expr, names, lr["receptor"], "min")
    keep, sim = scriabin.lr_similarity(scriabin.ccim(lig, rec))
    labels = lr["lr_name"].to_numpy()[keep]
    return pd.DataFrame(sim, index=labels, columns=labels)


def _tree_spread(sim: pd.DataFrame, power: int) -> tuple[float, float]:
    """Range of the dendrogram's merge heights and their median gap."""
    tree = _wgcna.hclust_complete(1.0 - _wgcna.signed_tom(sim.to_numpy() ** power))
    heights = np.sort(tree.height)
    return float(heights[-1] - heights[0]), float(np.median(np.diff(heights)))


def _max_change(a: pd.DataFrame, b: pd.DataFrame) -> float:
    shared = a.index.intersection(b.index)
    diff = a.loc[shared, shared].to_numpy() - b.loc[shared, shared].to_numpy()
    return float(np.nanmax(np.abs(diff)))


def main() -> None:
    d, rout = Path(sys.argv[1]), Path(sys.argv[2])
    drops = [-1, *map(int, sys.argv[3:])]
    X = scipy.io.mmread(d / "expr.mtx").T.tocsr()  # cells x features
    names = pd.Index(pd.read_csv(d / "features.csv", header=None)[0].astype(str))
    meta = pd.read_csv(d / "meta.csv")
    lr = pd.read_csv(d / "lr.csv")
    rows = []
    for s in meta["sample"].unique():
        cells = np.flatnonzero(meta["sample"].to_numpy() == s)
        base = None
        for drop in drops:
            res = scriabin.modules_from_similarity(
                sim := _similarity(
                    X[np.delete(cells, drop) if drop >= 0 else cells], lr, names
                )
            )
            r_mods = json.loads((rout / f"{s}_drop{drop}.json").read_text())
            r_mods = {k: v if isinstance(v, list) else [v] for k, v in r_mods.items()}
            if base is None:
                base = (res.modules, r_mods, sim)
                spread, gap = _tree_spread(sim, res.soft_power)
            rows.append(
                {
                    "sample": s,
                    "n_cells": len(cells),
                    "dropped": drop,
                    "R == Python": res.modules == r_mods,
                    "soft_power": res.soft_power,
                    "n_modules": len(res.modules),
                    "stability_python": module_stability(base[0], res.modules),
                    "stability_r": module_stability(base[1], r_mods),
                    "max_similarity_change": _max_change(base[2], sim),
                    "height_range": spread,
                    "median_height_gap": gap,
                }
            )
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
