"""The canonical ALAD RISE (PARAFAC2) fit shared by this figure's panels."""

from functools import cache

import xarray as xr

from ....factors import load_factors_netcdf
from ....runners.datasets import DATASETS

DATASET = "alad"
XLABEL = "RISE component"


@cache
def rise_factors(dataset: str = DATASET) -> xr.Dataset:
    return load_factors_netcdf(DATASETS[dataset].factors_path())
