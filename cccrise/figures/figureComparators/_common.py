"""Shared inputs of the comparator figure: the ALAD comparator fits at their
own selected ranks on the coarse and fine annotations, and cccRISE's five
programs."""

from functools import cache

import numpy as np
import pandas as pd

from ...benchmark import ComparatorFit, read_fits
from ...factors import load_factors_netcdf
from ...runners.datasets import DATASETS, VALIDATION_DIR
from .. import common

DATASET = "alad"
METHODS = ("Pseudobulk CPD", "Tensor-cell2cell", "STACCato")
SHORT = {"Pseudobulk CPD": "PB", "Tensor-cell2cell": "TC2C", "STACCato": "STAC"}
#: Annotation -> output suffix and the `obs` column holding its labels.
ANNOTATIONS = {"coarse": ("", "broad_cell_type"), "fine": ("_fine", "celltype")}
FITS = [(method, annotation) for annotation in ANNOTATIONS for method in METHODS]
#: cccRISE's programs, named by their biology (Figs. 2 and 3).
PROGRAMS = {
    1: "C1 CEACAM, 1 patient",
    2: "C2 XCL–XCR1, DC",
    3: "C3 CCL17/22–CCR4, DC",
    4: "C4 cytotoxic, CLAD",
    5: "C5 BTC–EGFR, repair",
}


def fit_label(method: str, annotation: str) -> str:
    return f"{SHORT.get(method, method)} {annotation}"


def validation(name: str, annotation: str = "coarse") -> pd.DataFrame:
    """A runner's ALAD table (`<name>_alad<suffix>.parquet`) for `annotation`."""
    suffix = ANNOTATIONS[annotation][0]
    return common.load_runner_output(f"{name}_{DATASET}{suffix}.parquet", VALIDATION_DIR)


@cache
def own_fits(annotation: str) -> dict[str, ComparatorFit]:
    """Each comparator's fit at its own selected rank, by method."""
    path = VALIDATION_DIR / DATASETS[DATASET].benchmark_fits_filename(ANNOTATIONS[annotation][0])
    fits, _info, _conditions = read_fits(path)
    return {fit.method: fit for fit in fits if fit.rank_mode == "own"}


@cache
def cccrise_lr() -> np.ndarray:
    """cccRISE's LR factor (pairs x components)."""
    return np.asarray(load_factors_netcdf(DATASETS[DATASET].factors_path())["cp_D"].values)


def effective_pairs(lr: np.ndarray) -> np.ndarray:
    """Per column, the inverse Simpson index of its squared loadings: the
    number of equally weighted pairs that would carry the same concentration."""
    share = lr**2 / np.sum(lr**2, axis=0, keepdims=True)
    return 1.0 / np.sum(share**2, axis=0)
