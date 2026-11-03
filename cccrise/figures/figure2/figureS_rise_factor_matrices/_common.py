"""The canonical ALAD RISE (PARAFAC2) fit shared by this figure's panels."""

from functools import cache

import numpy as np
import pandas as pd
import xarray as xr

from ....factors import load_factors_netcdf
from ....runners.datasets import DATASETS

DATASET = "alad"
XLABEL = "RISE component"


def odd_component_ticks(ax, n: int) -> None:
    """Label every other component, as 17 crowd a CP-sized matrix's axis."""
    ax.set_xticks(range(0, n, 2), range(1, n + 1, 2))


@cache
def rise_factors() -> xr.Dataset:
    return load_factors_netcdf(DATASETS[DATASET].factors_path())


def normalized_table(labels, loadings: np.ndarray, label: str) -> pd.DataFrame:
    """Factor `loadings` (rows labeled `labels`, in the plotted order) as
    drawn: each component divided by its largest magnitude, keeping sign."""
    columns = [f"{XLABEL} {c}" for c in range(1, loadings.shape[1] + 1)]
    table = pd.DataFrame(loadings / np.abs(loadings).max(axis=0), columns=columns)
    table.insert(0, label, labels)
    return table
