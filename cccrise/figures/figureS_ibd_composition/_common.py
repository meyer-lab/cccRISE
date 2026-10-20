"""Inputs of the IBD composition supplement (``run_ibd_composition``)."""

from functools import cache

import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from ...runners.provenance import read_metadata
from .. import common, style

DATASET = "ibd"
HINT = "    uv run python -m cccrise.runners.run_ibd_composition --dataset ibd --max-cells-per-condition 2000 --cell-cap-seed 0"
METHODS = ("cccRISE", "Pseudobulk CPD", "Tensor-cell2cell", "STACCato")
SHORT = {"cccRISE": "cccRISE", "Pseudobulk CPD": "PB", "Tensor-cell2cell": "TC2C", "STACCato": "STAC"}
#: The fine types carrying cccRISE's IBD programs (Fig. 4), labeled where drawn.
CARRIERS = ("Enteroendocrine cells", "Enterochromaffin cells", "Tuft cells", "Monocytes SOD2+")


@cache
def table(name: str) -> pd.DataFrame:
    return common.load_runner_output(f"{name}_{DATASET}.parquet", VALIDATION_DIR, hint=HINT)


def aurocs() -> dict:
    return read_metadata(VALIDATION_DIR / f"composition_celltypes_{DATASET}.parquet")[
        "auroc_inflamed_vs_healthy"
    ]


def lineage_colors(lineages) -> dict[str, str]:
    return dict(zip(sorted(set(lineages)), style.CATEGORICAL, strict=False))
