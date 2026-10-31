"""The canonical RISE (PARAFAC2) fit of a dataset, shared by this figure's
panels and their IBD counterparts (`figure5.figureS_ibd_rise_factor_matrices`)."""

from functools import cache

import numpy as np
import pandas as pd
import xarray as xr

from ....factors import load_factors_netcdf
from ....runners.datasets import DATASETS

DATASET = "alad"
XLABEL = "RISE component"


@cache
def rise_factors(dataset: str = DATASET) -> xr.Dataset:
    return load_factors_netcdf(DATASETS[dataset].factors_path())


def normalized_table(labels, loadings: np.ndarray, label: str) -> pd.DataFrame:
    """Factor `loadings` (rows labeled `labels`, in the plotted order) as
    drawn: each component divided by its largest magnitude, keeping sign."""
    columns = [f"{XLABEL} {c}" for c in range(1, loadings.shape[1] + 1)]
    table = pd.DataFrame(loadings / np.abs(loadings).max(axis=0), columns=columns)
    table.insert(0, label, labels)
    return table
