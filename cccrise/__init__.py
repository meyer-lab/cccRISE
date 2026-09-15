"""Cell communication analysis using PARAFAC2 (cccRISE)."""

import os
from pathlib import Path

# Ensure CUDA headers are found by CuPy if present
if "CUDA_PATH" not in os.environ and Path("/usr/include/cuda.h").exists():
    os.environ["CUDA_PATH"] = "/usr"

from .anova import (
    adjust_pvalues,
    anova_component_associations,
    prepare_anova_inputs,
    prepare_group_labels,
    run_multiple_factors_anova,
    run_saved_factors_anova,
)
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
    "adjust_pvalues",
    "anova_component_associations",
    "clean_memory",
    "factors_to_xarray",
    "load_factors_netcdf",
    "load_factors_to_anndata",
    "load_or_compute_dataframe",
    "prepare_anova_inputs",
    "prepare_group_labels",
    "run_ccc_rise_workflow",
    "run_multiple_factors_anova",
    "run_pseudobulk_workflow",
    "run_saved_factors_anova",
    "save_factors_netcdf",
]
