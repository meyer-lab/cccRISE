"""Cell communication analysis using PARAFAC2 (cccRISE)."""

import os
from pathlib import Path

# Ensure CUDA headers are found by CuPy if present
if "CUDA_PATH" not in os.environ and Path("/usr/include/cuda.h").exists():
    os.environ["CUDA_PATH"] = "/usr"

from .factors import (
    ALAD_FACTORS_PATH,
    COVID19_FACTORS_PATH,
    clean_memory,
    factors_to_xarray,
    load_factors_netcdf,
    load_factors_to_anndata,
    load_or_compute_dataframe,
    save_factors_netcdf,
)
from .tensor import run_ccc_rise_workflow, run_pseudobulk_workflow

__all__ = [
    "ALAD_FACTORS_PATH",
    "COVID19_FACTORS_PATH",
    "clean_memory",
    "factors_to_xarray",
    "load_factors_netcdf",
    "load_factors_to_anndata",
    "load_or_compute_dataframe",
    "run_ccc_rise_workflow",
    "run_pseudobulk_workflow",
    "save_factors_netcdf",
]
