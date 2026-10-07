"""The canonical ALAD RISE (PARAFAC2) fit shared by this figure's panels."""

from functools import cache

import xarray as xr

from ....factors import load_factors_netcdf
from ....runners.datasets import DATASETS

DATASET = "alad"
XLABEL = "RISE component"
TICKS = (1, 5, 10, 15)  # component ticks on narrow heatmaps


@cache
def rise_factors() -> xr.Dataset:
    return load_factors_netcdf(DATASETS[DATASET].factors_path())
