"""Patient-level bootstrap of a tensor comparator's component--phenotype ANOVA.

The comparator counterpart of `anova_stability`: resample the patient roster
(`utils.resample_conditions`, stratified, paired samples carried together),
rebuild the cell-type pseudobulk tensor on the resampled cells, refit the
comparator at its fixed rank, run the same ANOVA (`anova.prepare_anova_inputs`
then `anova.anova_component_associations`), and match each replicate's
components back to the full-cohort fit on the ligand-receptor mode
(`anova_stability.align_to_reference_components`). Each fit's condition
factor is oriented to a positive column mean (with its LR column flipped
alongside, which leaves the CP model unchanged), as `standardize_cc_pf2`
does for cccRISE, so `GroupMeanDiff` keeps its direction across fits.
"""

from collections.abc import Callable
from typing import Any

import anndata
import numpy as np
import pandas as pd

from ..anova import anova_component_associations, prepare_anova_inputs
from ..benchmark import pseudobulk_cpd_fit, pseudobulk_tensor, tensor_cell2cell_fit
from ..import_data import add_cond_idxs
from ..utils import resample_conditions
from .anova_stability import REPLICATE_COLUMNS, align_to_reference_components

#: Factor order is (condition, sender, receiver, lr); match on lr, as cccRISE.
LR_MODE = 3


def comparator_fitter(method: str, backend: str = "numpy") -> Callable:
    """`fit(tensor, rank, seed) -> factors` for a CP comparator."""
    if method == "Pseudobulk CPD":
        return lambda tensor, rank, seed: pseudobulk_cpd_fit(tensor, rank, seed)[1]
    if method == "Tensor-cell2cell":
        return lambda tensor, rank, seed: tensor_cell2cell_fit(
            tensor, rank, seed, backend
        )[1]
    raise ValueError(f"No bootstrap fitter for {method!r}")


def orient(factors: list[np.ndarray]) -> list[np.ndarray]:
    """Flip components so each condition column has a nonnegative mean,
    flipping the LR column with it."""
    signs = np.where(factors[0].mean(axis=0) < 0, -1.0, 1.0)
    out = [f.copy() for f in factors]
    out[0] *= signs
    out[LR_MODE] *= signs
    return out


def fit_and_test(
    adata: anndata.AnnData,
    fit: Callable,
    rank: int,
    seed: int,
    columns: dict[str, str],
    lr_pairs: pd.DataFrame,
) -> dict[str, Any]:
    """Pseudobulk tensor, comparator fit and ANOVA on `adata`.

    `columns` names the 'condition', 'celltype' and 'group' obs columns."""
    adata = add_cond_idxs(adata, columns["condition"])
    tensor, _types, pairs = pseudobulk_tensor(
        adata, columns["condition"], columns["celltype"], lr_pairs
    )
    factors = orient(fit(tensor, rank, seed))
    X, groups = prepare_anova_inputs(
        adata, factors, columns["condition"], columns["group"]
    )
    return {
        "factors": factors,
        "lr_symbols": pairs["interaction_symbol"].to_numpy(),
        "anova": anova_component_associations(X, groups),
    }


def comparator_replicate(
    adata: anndata.AnnData,
    reference: dict[str, Any],
    fit: Callable,
    rank: int,
    seed: int,
    columns: dict[str, str],
    lr_pairs: pd.DataFrame,
    patient_column: str | None = None,
) -> pd.DataFrame:
    """One replicate, in `anova_stability.REPLICATE_COLUMNS` (R2XCP is NaN)."""
    resampled = resample_conditions(
        adata,
        columns["condition"],
        stratify_by=columns["group"],
        patient_by=patient_column,
        random_seed=seed,
    )
    result = fit_and_test(resampled, fit, rank, seed, columns, lr_pairs)
    if not np.array_equal(result["lr_symbols"], reference["lr_symbols"]):
        raise ValueError("Replicate and reference kept different LR pairs")
    anova = result["anova"]
    aligned = align_to_reference_components(
        reference["factors"],
        result["factors"],
        anova["GroupMeanDiff"].to_numpy(float),
        modes=(LR_MODE,),
    ).rename(columns={"Value": "GroupMeanDiff"})
    matched = aligned["MatchedComponent"].to_numpy(int) - 1
    for column in ("EtaSquared", "F", "P"):
        aligned[column] = anova[column].to_numpy(float)[matched]
    aligned["R2XCP"] = np.nan
    return aligned[REPLICATE_COLUMNS]
