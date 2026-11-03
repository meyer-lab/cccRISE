"""Write the shared inputs for the Scriabin / NICHES R-vs-Python checks.

Two samples of a registry dataset (by default the first two, in name order,
with 700-1,000 cells, so Scriabin averages two 300-cell subsamples each), on
cccRISE's normalized expression, the fit's genes and its LR pairs. Complexes
are written as min-aggregated pseudo-genes (Scriabin side) and as
``_``-joined subunits (NICHES side, which multiplies them itself).

Usage: uv run python rcheck/export_inputs.py OUT [--dataset ibd] [SAMPLE ...]
(named samples instead of the default two, e.g. for `cell_drop.R`)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from _common import dataset_inputs

from cccrise.comparators import scriabin
from cccrise.comparators.inputs import with_complex_pseudogenes

SEED = 20261006


def default_samples(sizes: pd.Series) -> list[str]:
    """The first two samples, in name order, that average two subsamples."""
    two_reps = sizes[(sizes >= 700) & (sizes <= 1000)].sort_index()
    return list(two_reps.index[:2])


def main(out: Path, dataset: str, samples: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    inp = dataset_inputs(dataset)
    sizes = inp.obs["sample"].value_counts()
    samples = samples or default_samples(sizes)
    rows = np.flatnonzero(inp.obs["sample"].isin(samples).to_numpy())

    X, names = with_complex_pseudogenes(inp.X[rows], inp.genes, inp.lr, agg="min")
    bad = [n for n in names if "_" in n or "=" in n]
    assert not bad, f"feature names Seurat would rewrite: {bad[:5]}"
    scipy.io.mmwrite(out / "expr.mtx", X.T.tocoo())  # genes x cells
    pd.Series(names).to_csv(out / "features.csv", index=False, header=False)
    meta = inp.obs.iloc[rows].rename_axis("cell").reset_index()
    meta.to_csv(out / "meta.csv", index=False)
    inp.lr.assign(
        ligand_niches=inp.lr["ligand"].str.replace("&", "_"),
        receptor_niches=inp.lr["receptor"].str.replace("&", "_"),
    ).to_csv(out / "lr.csv", index=False)

    # Scriabin subsamples: Python's stratified draws, fed to R through a
    # `sample_n` shim (per type, in call order).
    rng = np.random.default_rng(SEED)
    draws = {}
    for s in samples:
        m = meta[meta["sample"] == s].reset_index(drop=True)
        n_rep = scriabin.n_iterations(len(m))
        reps = [
            m["cell"].to_numpy()[
                scriabin.stratified_subsample(m["celltype"].to_numpy(), rng)
            ]
            for _ in range(n_rep)
        ]
        draws[s] = [list(map(str, r)) for r in reps]
    (out / "subsamples.json").write_text(json.dumps(draws))
    (out / "dataset.txt").write_text(dataset + "\n")
    print(
        f"{dataset}: samples {samples} sizes {[int(sizes[s]) for s in samples]}; "
        f"features {len(names)}; LR {len(inp.lr)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("samples", nargs="*")
    parser.add_argument("--dataset", default="alad", choices=["alad", "ibd"])
    args = parser.parse_args()
    main(args.out, args.dataset, args.samples)
