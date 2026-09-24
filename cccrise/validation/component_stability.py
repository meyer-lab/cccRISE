"""
Component stability of the CCC-RISE pipeline at fixed RISE and CP ranks.

Everything here is *conditional on* ranks that have already been chosen (see
`rank_selection` for the rules that choose them). The goal is to evaluate
whether the identified components recovered at those fixed ranks are stable and
reproducible under data resampling -- directly addressing Reviewer 2's major
comment #4.

Distinction between Per-Fit Diagnostics and Cross-Fit Stability Metrics
----------------------------------------------------------------------
This module computes and tracks two distinct categories of metrics:

1. **Per-Fit Diagnostics (Single-Fit Properties)**:
   Diagnostics evaluating the reconstruction quality, identifiability, and
   numerical conditioning of an *individual* model fit on a specific dataset or
   resampled tensor:
   - `R2X`: Variance explained (reconstruction quality) of the RISE or CPD fit.
   - `CoreConsistency`: Tucker core consistency (CORCONDIA) of the CPD fit.
   - `MaxModesCollinear` & `WorstPairCanceling`: Collinearity and degenerate
     canceling pairs between CP component vectors within a fit.
   These diagnostics are replicate-level scalars that characterize the fit
   itself, allowing users to verify that replicate factorizations converged
   to valid, non-degenerate solutions.

2. **Cross-Fit Stability Metrics (Comparative Properties)**:
   Comparative metrics evaluating how reproducibly the factor vectors align
   between a reference fit (e.g. on the full cohort) and a replicate fit across
   data perturbations:
   - `FMS` (Factor Match Score): The congruence product across compared modes
     of matched component vectors between reference and replicate factorizations,
     assigned via optimal (Hungarian) bipartite matching.
   - `RunnerUpComponent` & `RunnerUpFMS`: The next-best alternative candidate
     and its score, diagnosing component identifiability and match ambiguity.
   - `SignedCosMode0`..`SignedCosMode3`: Per-mode signed cosine similarities
     between matched component vectors, capturing directional consistency.

Mode Hold-Outs Across Resamplings
---------------------------------
- **RISE (Mode 1 held out)**: Mode 0 (Conditions $A$) and Mode 2 (Genes $C$) are
  compared; Mode 1 (Basis $B$, $r \\times r$) is held out (`RISE_SKIP_MODES = (1,)`).
- **CPD (Modes 1 and 2 held out)**: Mode 0 (Conditions $G$) and Mode 3
  (Ligand-receptor pairs $F$) are compared; Modes 1 and 2 (Sender $D$ and Receiver
  $E$ eigen-state loadings) are held out (`CPD_SKIP_MODES = (1, 2)`) because
  eigen-state indexing across bootstrap fits is not permuted into CPD rows.

Row Alignment
-------------
- When conditions are resampled with replacement (`conditions` axis), reference
  mode-0 rows ($A$ and $G$) are expanded via `reference_condition_rows` to align
  with the replicate's resampled condition roster.
- When ligand-receptor pairs are resampled with replacement (`lr_pairs` axis),
  reference mode-3 rows ($F$) are expanded using the drawn LR indices to align
  with the replicate's resampled LR roster.
- The seed axis (`SEED_AXIS`) needs no alignment at all: the roster is the
  reference's own, so mode 0 already matches row for row. That is what makes
  its FMS comparable to the reference's on a per-component basis.
"""

from collections.abc import Callable
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tensorly.decomposition import parafac

from ..ccc import calculate_r2x
from ..ccc_rise import standardize_cc_pf2, validate_cp_rank
from ..factors import clean_memory
from ..import_data import add_cond_idxs
from ..rank_selection import (
    candelinc_compress,
    candelinc_r2x,
    cp_core_consistency,
    cpd_component_collinearity,
)
from ..tensor import (
    calculate_interaction_tensor,
    rise_store_r2x,
)
from ..utils import (
    replicate_seed,
    resample_cells,
    resample_conditions,
    resample_tensor,
)

Factors = list[np.ndarray]

#: Values of the 'Stage' column.
STAGE_RISE = "RISE"
STAGE_CPD = "CPD"

#: Modes held out during factor match score (FMS) calculations.
RISE_SKIP_MODES: tuple[int, ...] = (1,)
CPD_SKIP_MODES: tuple[int, ...] = (1, 2)

#: Name of the reproducibility-ceiling axis: identical data, a different random
#: seed, refit end to end. `random_state` genuinely moves both stages: `parafac2_nd`
#: seeds the Gaussian projection of its randomized-SVD initialization and its
#: compression stage, and tensorly's ``parafac(init="svd", svd="randomized_svd")`` does
#: the same, so the axis is not degenerate.
SEED_AXIS = "seed_only"

#: Percentile of the seed axis's FMS taken as the reproducibility band's lower
#: edge.
SEED_BAND_PERCENTILE = 5.0

#: Axes that refit RISE.
RISE_REFIT_AXES = ("cells", "conditions", SEED_AXIS)

#: Axes run by default.
DEFAULT_AXES = ("cells", "conditions", "lr_pairs", SEED_AXIS)

#: How an axis prepares the data it is fit on. Each entry takes
#: ``(adata, condition_column, group_column, patient_column, seed)``.
#:
#: `SEED_AXIS`'s entry perturbs nothing -- its perturbation is applied
#: downstream, by handing `fit_stability_pipeline` a different `random_state`.
RESAMPLERS: dict[
    str,
    Callable[[anndata.AnnData, str, str | None, str | None, int], anndata.AnnData],
] = {
    "cells": lambda adata, col, group, patient, seed: resample_cells(
        adata, condition_name=col, random_seed=seed
    ),
    "conditions": lambda adata, col, group, patient, seed: resample_conditions(
        adata,
        condition_name=col,
        stratify_by=group,
        random_seed=seed,
        patient_by=patient,
    ),
    SEED_AXIS: lambda adata, col, group, patient, seed: adata.copy(),
}


# =============================================================================
# Matching two fits' components
# =============================================================================


def per_component_fms(
    factors_a: Factors,
    factors_b: Factors,
    skip_modes: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Factor match score decomposed into its per-component terms.

    Computes the congruence product the FMS definition uses -- the product
    across modes of the cosine between matched component vectors -- assigns
    components by optimal (Hungarian) matching on that product, and returns one
    row per matched pair. The mean of the returned 'FMS' column *is* the
    aggregate FMS, so this is a decomposition of the number the Methods already
    define rather than a different metric.

    Weights are ignored (equivalently, ``consider_weights=False``), which is
    forced whenever a mode is skipped: dropping a mode leaves the scaling split
    between the remaining modes indeterminate.

    Matching is by congruence rather than by component index because
    `ccc_rise.standardize_cc_pf2` orders components by ``var(A)/mean(A)``, a
    ratio whose denominator can approach zero, so the index itself moves under
    resampling even when the components do not.

    Parameters
    ----------
    factors_a, factors_b : list[np.ndarray]
        Factor lists from two fits of equal rank, sharing dimensions in every
        mode that is not skipped. `factors_a` supplies the reference numbering.
    skip_modes : tuple[int, ...], default=()
        Mode indices to exclude; see `RISE_SKIP_MODES` and `CPD_SKIP_MODES`.

    Returns
    -------
    pd.DataFrame
        One row per component of `factors_a`, with columns:

        - 'Component', 'MatchedComponent': 1-based indices into `factors_a`
          and `factors_b`.
        - 'FMS': that component's congruence product over the compared modes.
        - 'RunnerUpComponent', 'RunnerUpFMS': the next-best candidate in
          `factors_b` and its score, or -1 / NaN at rank 1. A runner-up close
          to 'FMS' means the replicate could not tell the two apart, which is
          a different failure from the component having moved.
        - 'SignedCosMode0'..'SignedCosMode3': the signed cosine in each
          compared mode, NaN in skipped modes. The product's sign is not
          recoverable from 'FMS', which is absolute, so a genuine sign
          reversal stays visible.
    """
    n_modes = len(factors_a)
    signed = {}
    congruence = np.ones((factors_a[0].shape[1], factors_b[0].shape[1]))
    for mode, (fa, fb) in enumerate(zip(factors_a, factors_b, strict=True)):
        if mode in skip_modes:
            continue
        na = fa / np.linalg.norm(fa, axis=0, keepdims=True)
        nb = fb / np.linalg.norm(fb, axis=0, keepdims=True)
        signed[mode] = na.T @ nb
        congruence = congruence * signed[mode]

    congruence = np.abs(congruence)
    row_ind, col_ind = linear_sum_assignment(-congruence)

    # Best alternative for each reference component, excluding its assignment.
    masked = congruence.copy()
    masked[row_ind, col_ind] = -np.inf
    if masked.shape[1] > 1:
        runner_up_col = masked[row_ind].argmax(axis=1)
        runner_up_fms = masked[row_ind, runner_up_col]
        runner_up_component = runner_up_col + 1
    else:
        runner_up_fms = np.full(row_ind.size, np.nan)
        runner_up_component = np.full(row_ind.size, -1)

    out = pd.DataFrame(
        {
            "Component": row_ind + 1,
            "MatchedComponent": col_ind + 1,
            "FMS": congruence[row_ind, col_ind],
            "RunnerUpComponent": runner_up_component,
            "RunnerUpFMS": runner_up_fms,
        }
    )
    for mode in range(max(n_modes, 4)):
        out[f"SignedCosMode{mode}"] = (
            signed[mode][row_ind, col_ind] if mode in signed else np.nan
        )
    return out


def reference_condition_rows(
    reference_labels: np.ndarray, replicate_labels: np.ndarray
) -> np.ndarray:
    """Reference condition-mode rows corresponding to a replicate's roster.

    RISE's `A` and CP's `G` index conditions in the order
    `import_data.add_cond_idxs` assigns, namely sorted unique labels, so a
    replicate's row *r* belongs to whichever condition it was drawn from.
    `utils.resample_conditions` suffixes repeat draws (``__rep2``, ...); the
    suffix is stripped here to recover the source label.

    The returned index expands the reference's rows onto the replicate's
    roster, so the condition mode can be *compared* rather than skipped. That
    matters most on the axis that resamples conditions: dropping mode 0 there
    would leave RISE stability resting on the gene factor alone, precisely on
    the axis that perturbs the cohort, and would put the two axes' FMS values
    on different scales. A condition drawn twice contributes twice on both
    sides, which weights conditions exactly as the fit itself did.

    Parameters
    ----------
    reference_labels : np.ndarray
        The reference fit's condition labels, aligned to its mode-0 rows.
    replicate_labels : np.ndarray
        The replicate's condition labels, aligned to its mode-0 rows.

    Returns
    -------
    np.ndarray
        Integer index into `reference_labels`, of length
        ``len(replicate_labels)``.

    Raises
    ------
    ValueError
        If a replicate label has no source row in the reference, which would
        mean the two fits' condition modes are not relatable at all.
    """
    position = {str(label): i for i, label in enumerate(reference_labels)}
    rows = []
    for label in replicate_labels:
        source = str(label).split("__rep")[0]
        if source not in position:
            raise ValueError(
                f"Replicate condition {label!r} resolves to source condition "
                f"{source!r}, which is absent from the reference fit's "
                "condition mode, so the two fits cannot be compared on it."
            )
        rows.append(position[source])
    return np.asarray(rows, dtype=int)


def _expand_reference_mode(factors: Factors, mode: int, rows: np.ndarray) -> Factors:
    """`factors` with specified mode rows re-indexed onto a replicate's roster."""
    return [factors[m][rows] if m == mode else factors[m] for m in range(len(factors))]


# =============================================================================
# One pipeline pass
# =============================================================================


def _condition_labels(adata: anndata.AnnData, condition_column: str) -> np.ndarray:
    """Condition labels aligned to the fit's mode-0 rows.

    Read back out of the `condition_unique_idxs` column `add_cond_idxs` wrote,
    rather than re-derived with an independent `np.unique`, so the labels
    cannot disagree with the rows they name. The two orderings differ whenever
    the condition column is not already a string: `add_cond_idxs` sorts in the
    column's native dtype, so an integer column gives rows in `[1, 2, 10]`
    order while a string sort of the same labels gives `['1', '10', '2']`.
    Consumers -- `reference_condition_rows` above all -- treat this array as
    positionally aligned to mode 0, and a silent disagreement would have
    `per_component_fms` compare the wrong patients and still return a
    plausible number.
    """
    cond_idx = np.asarray(adata.obs["condition_unique_idxs"]).astype(int)
    labels = np.asarray(adata.obs[condition_column]).astype(str)
    order = np.empty(cond_idx.max() + 1, dtype=object)
    order[cond_idx] = labels
    assert len(order) == adata.uns["Pf2_A"].shape[0], (
        f"{len(order)} condition labels for "
        f"{adata.uns['Pf2_A'].shape[0]} rows of Pf2_A."
    )
    return order


def fit_stability_pipeline(
    adata: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    condition_column: str,
    rise_rank: int,
    cp_rank: int,
    group_column: str | None = None,
    rise_n_iter_max: int = 100,
    cp_n_iter_max: int = 100,
    init: str = "svd",
    svd: str = "randomized_svd",
    complex_sep: str | None = "&",
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
    random_state: int = 0,
    keep_tensor: bool = False,
    backend: str | None = None,
    compress: bool = True,
    oversample: int = 5,
    *,
    rise_tol: float,
    cp_tol: float,
    fix_B_identity: bool = False,
) -> dict[str, Any]:
    """Run RISE, build the communication tensor, and fit CP over `adata`.

    `adata` is written into by `parafac2.store_pf2`, so callers must hand
    over a copy they do not need pristine -- which is what the resamplers
    return.

    Parameters
    ----------
    adata : anndata.AnnData
        Data to fit: the full cohort for the reference pass, a resampled
        copy for a replicate.
    lr_pairs : pd.DataFrame
        Ligand-receptor pairs to score.
    condition_column : str
        Column in `obs` holding the condition (patient) identity.
    rise_rank, cp_rank : int
        Ranks for the RISE and CP stages.
    rise_n_iter_max, rise_tol : int, float
        Fitting controls for RISE.
    cp_n_iter_max, cp_tol, init, svd
        Fitting controls for CP.
    complex_sep, communication_score, hill_k
        Communication-tensor options; see `ccc_rise.calc_communication_score`.
    random_state : int, default=0
        Seed for the RISE fit and the CP fit.
    keep_tensor : bool, default=False
        Retain the communication tensor on the returned fit. Only the
        reference pass needs it.
    backend : str | None, default=None
        Backend the RISE fit runs on ('cpu', 'cupy', 'mlx', or None to
        auto-detect); passed straight through to `tensor.rise_store_r2x`.
    compress : bool, default=True
        Fit CP inside a CANDELINC core sized for `cp_rank` instead of the
        full tensor (see `validation.component_stability._fit_cp`).
    oversample : int, default=5
        Extra basis columns per mode beyond `cp_rank`, when `compress`.

    Returns
    -------
    dict[str, Any]
        Fitted factors, diagnostics, and metadata.
    """
    adata = add_cond_idxs(adata, condition_column)
    adata, r2x_rise = rise_store_r2x(
        adata,
        rank=rise_rank,
        n_iter_max=rise_n_iter_max,
        tolerance=rise_tol,
        random_state=random_state,
        backend=backend,
        fix_B_identity=fix_B_identity,
    )
    tensor, filtered_lr_pairs = calculate_interaction_tensor(
        adata,
        lr_pairs,
        complex_sep=complex_sep,
        communication_score=communication_score,
        hill_k=hill_k,
    )
    validate_cp_rank(cp_rank, tensor.shape)

    cp_weights, cp_factors, r2x_cp, diagnostics = _fit_cp(
        tensor,
        cp_rank,
        n_iter_max=cp_n_iter_max,
        tol=cp_tol,
        init=init,
        svd=svd,
        random_state=random_state,
        compress=compress,
        oversample=oversample,
    )
    cp_weights, cp_factors = standardize_cc_pf2(cp_weights, cp_factors)

    return {
        "rise_factors": [
            adata.uns["Pf2_A"],
            adata.uns["Pf2_B"],
            np.asarray(adata.varm["Pf2_C"]),
        ],
        "cp_weights": cp_weights,
        "cp_factors": cp_factors,
        "condition_labels": _condition_labels(adata, condition_column),
        "gene_names": np.asarray(adata.var_names),
        "lr_symbols": filtered_lr_pairs["interaction_symbol"].to_numpy(),
        "tensor": tensor if keep_tensor else None,
        "r2x_rise": r2x_rise,
        "r2x_cp": r2x_cp,
        "cpd_diagnostics": diagnostics,
    }


# =============================================================================
# The shared replicate loop
# =============================================================================


def _skip_label(skip_modes: tuple[int, ...]) -> str:
    """The 'SkippedModes' column value, e.g. ``(1, 2)`` -> ``"1,2"``."""
    return ",".join(str(mode) for mode in skip_modes)


def _cpd_diagnostics(
    tensor: np.ndarray, weights: np.ndarray, factors: Factors, rank: int
) -> dict[str, Any]:
    """Replicate-level CP fit diagnostics at the fixed rank.

    Reported beside the FMS values so that a wide spread traceable to a poor
    or degenerate fit is distinguishable from genuine component instability.
    """
    collinearity = cpd_component_collinearity({rank: (weights, factors)}).iloc[0]
    return {
        "CoreConsistency": cp_core_consistency(tensor, weights, factors),
        "MaxModesCollinear": int(collinearity["MaxModesCollinear"]),
        "WorstPairCanceling": collinearity["WorstPairCanceling"],
    }


def _fit_cp(
    tensor: np.ndarray,
    rank: int,
    n_iter_max: int,
    tol: float,
    init: str,
    svd: str,
    random_state: int | None,
    compress: bool = True,
    oversample: int = 5,
) -> tuple[np.ndarray, Factors, float, dict[str, Any]]:
    """Fit CP at a fixed rank, in a CANDELINC core by default.

    `compress` fits against a core sized for `rank`
    (`rank_selection.candelinc_compress`) instead of the full tensor, then
    maps the factors back onto the tensor's own modes with ``A_n = U_n F_n``.
    R2X (via `rank_selection.candelinc_r2x`), core consistency and
    collinearity are each proven invariant to that substitution, so this is a
    speed optimization -- what a replicate's diagnostics and FMS say does not
    change, only how fast they are computed.

    Returns weights, factors (in `tensor`'s own space, ready for
    `per_component_fms`), R2X against `tensor`, and the `_cpd_diagnostics`
    dict -- each computed in the (cheaper) core space when compressed.
    """
    fit_target = tensor
    bases: Factors | None = None
    lost_sq = 0.0
    if compress:
        fit_target, bases, lost_sq = candelinc_compress(
            tensor, max_rank=rank, oversample=oversample
        )

    weights, factors = parafac(
        fit_target,
        rank,
        n_iter_max=n_iter_max,
        tol=tol,
        init=init,
        svd=svd,
        normalize_factors=True,
        random_state=random_state,
    )
    diagnostics = _cpd_diagnostics(fit_target, weights, factors, rank)
    r2x = candelinc_r2x(
        calculate_r2x(weights, factors, fit_target), fit_target, lost_sq
    )

    if bases is not None:
        factors = [b @ f for b, f in zip(bases, factors, strict=True)]

    return weights, factors, r2x, diagnostics


def _tag(
    block: pd.DataFrame,
    axis: str,
    replicate: int,
    seed: int,
    stage: str,
    r2x: float,
    n_cells: float,
    n_conditions: float,
    diagnostics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Prefix a `per_component_fms` block with its keys and per-fit diagnostics."""
    block = block.copy()
    block.insert(0, "Stage", stage)
    block.insert(0, "Seed", seed)
    block.insert(0, "Replicate", replicate)
    block.insert(0, "Axis", axis)
    block["R2X"] = r2x
    for column in ("CoreConsistency", "MaxModesCollinear", "WorstPairCanceling"):
        block[column] = (diagnostics or {}).get(column, np.nan)
    block["NCells"] = n_cells
    block["NConditions"] = n_conditions
    return block


def component_stability(
    adata: anndata.AnnData,
    reference: dict[str, Any],
    lr_pairs: pd.DataFrame,
    condition_column: str,
    rise_rank: int,
    cp_rank: int,
    group_column: str | None = None,
    patient_column: str | None = None,
    axes: tuple[str, ...] = DEFAULT_AXES,
    n_replicates: int = 20,
    rise_n_iter_max: int = 100,
    cp_n_iter_max: int = 100,
    init: str = "svd",
    svd: str = "randomized_svd",
    complex_sep: str | None = "&",
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
    random_state: int = 0,
    backend: str | None = None,
    compress: bool = True,
    oversample: int = 5,
    *,
    rise_tol: float,
    cp_tol: float,
    fix_B_identity: bool = False,
) -> pd.DataFrame:
    """Per-component stability of both stages under resampling, in long form.

    For each axis in `RISE_REFIT_AXES` a replicate prepares the data, refits
    RISE at `rise_rank`, and matches the resulting components onto the
    reference's with Mode 1 held out (`RISE_SKIP_MODES`); the communication
    tensor built from that same fit is then decomposed at `cp_rank` and
    matched in turn with Modes 1 and 2 held out (`CPD_SKIP_MODES`). 'cells'
    and 'conditions' resample; `SEED_AXIS` hands the pipeline the identical
    cohort and changes only the seed the fits start from, so it measures how
    far components move under the fitters' own randomness -- the
    reproducibility ceiling every resampling axis is read against, and the
    term that would otherwise be confounded with sampling variability.
    Because its roster is unchanged, its mode-0 rows line up with the
    reference's directly and no `reference_condition_rows` expansion applies.

    One axis acts on the reference tensor instead:
    - `lr_pairs`: The ligand-receptor mode bootstrapped with replacement,
      matching CPD modes 0 and 3 with reference mode-3 rows expanded by the
      drawn LR indices. It costs no PARAFAC2 fit and so emits CPD rows only.

    Parameters
    ----------
    adata : anndata.AnnData
        Full-cohort data to resample from.
    reference : dict[str, Any]
        The full-cohort fit, produced by ``fit_stability_pipeline(..., keep_tensor=True)``.
    lr_pairs : pd.DataFrame
        Ligand-receptor pairs to score.
    condition_column : str
        Column in `obs` holding the condition (patient) identity.
    rise_rank, cp_rank : int
        Ranks for the RISE and CP stages.
    group_column : str | None, default=None
        Condition-level phenotype column the condition draw is stratified within.
    patient_column : str | None, default=None
        Condition-level patient identity column, for the 'conditions' axis
        alone. When a patient contributes more than one condition, drawing one
        of them carries its sibling condition in and drops one other random
        draw to compensate (see `utils.resample_conditions`).
    axes : tuple[str, ...], default=DEFAULT_AXES
        Which axes to run.
    n_replicates : int, default=20
        Replicates per axis.
    rise_n_iter_max, rise_tol : int, float
        Fitting controls for RISE.
    cp_n_iter_max, cp_tol, init, svd
        Fitting controls for CP.
    complex_sep, communication_score, hill_k
        Communication-tensor options.
    random_state : int, default=0
        Base seed.
    backend : str | None, default=None
        Backend each replicate's RISE refit runs on; see
        `fit_stability_pipeline`.
    compress : bool, default=True
        Fit every replicate's CP stage inside a CANDELINC core; see
        `component_stability._fit_cp`.
    oversample : int, default=5
        Extra basis columns per mode beyond `cp_rank`, when `compress`.

    Returns
    -------
    pd.DataFrame
        Long-form table containing per-component FMS matches and per-fit
        diagnostics for each replicate.
    """
    unknown = set(axes) - set(RISE_REFIT_AXES) - {"lr_pairs"}
    if unknown:
        raise ValueError(f"Unknown stability axes: {sorted(unknown)}")
    if reference.get("tensor") is None and "lr_pairs" in axes:
        raise ValueError(
            "The 'lr_pairs' axis resamples the reference tensor, so the "
            "reference fit must be built with keep_tensor=True."
        )

    frames = []
    for axis in axes:
        for replicate in range(n_replicates):
            seed = replicate_seed(random_state, replicate)

            if axis in RISE_REFIT_AXES:
                resampled = RESAMPLERS[axis](
                    adata, condition_column, group_column, patient_column, seed
                )
                print(
                    f"   {axis} replicate {replicate + 1}/{n_replicates}: "
                    f"{resampled.n_obs:,} cells, "
                    f"{resampled.obs[condition_column].nunique()} conditions"
                )
                fit = fit_stability_pipeline(
                    resampled,
                    lr_pairs,
                    condition_column=condition_column,
                    rise_rank=rise_rank,
                    cp_rank=cp_rank,
                    rise_n_iter_max=rise_n_iter_max,
                    rise_tol=rise_tol,
                    fix_B_identity=fix_B_identity,
                    cp_n_iter_max=cp_n_iter_max,
                    cp_tol=cp_tol,
                    init=init,
                    svd=svd,
                    complex_sep=complex_sep,
                    communication_score=communication_score,
                    hill_k=hill_k,
                    random_state=seed,
                    backend=backend,
                    compress=compress,
                    oversample=oversample,
                )
                _check_rosters(reference, fit)

                if axis == "conditions":
                    rows = reference_condition_rows(
                        reference["condition_labels"], fit["condition_labels"]
                    )
                    ref_rise = _expand_reference_mode(
                        reference["rise_factors"], mode=0, rows=rows
                    )
                    ref_cp = _expand_reference_mode(
                        reference["cp_factors"], mode=0, rows=rows
                    )
                else:
                    ref_rise = reference["rise_factors"]
                    ref_cp = reference["cp_factors"]

                n_cells = float(resampled.n_obs)
                n_conditions = float(len(fit["condition_labels"]))
                frames.append(
                    _tag(
                        per_component_fms(
                            ref_rise, fit["rise_factors"], skip_modes=RISE_SKIP_MODES
                        ),
                        axis=axis,
                        replicate=replicate,
                        seed=seed,
                        stage=STAGE_RISE,
                        r2x=fit["r2x_rise"],
                        n_cells=n_cells,
                        n_conditions=n_conditions,
                    )
                )
                frames.append(
                    _tag(
                        per_component_fms(
                            ref_cp, fit["cp_factors"], skip_modes=CPD_SKIP_MODES
                        ),
                        axis=axis,
                        replicate=replicate,
                        seed=seed,
                        stage=STAGE_CPD,
                        r2x=fit["r2x_cp"],
                        n_cells=n_cells,
                        n_conditions=n_conditions,
                        diagnostics=fit["cpd_diagnostics"],
                    )
                )
                # CuPy's pool never returns freed blocks to the driver on its
                # own, so a long sweep of variably-sized resamples fragments
                # it until a fit spuriously trips the managed-memory fallback.
                clean_memory()
                continue

            # 'lr_pairs' is the only axis left that reaches the reference
            # tensor rather than refitting RISE; the guard above is exhaustive.
            tensor = cast(np.ndarray, reference["tensor"])
            perturbed, lr_indices = resample_tensor(
                tensor, random_seed=seed, return_indices=True
            )
            ref_cp = _expand_reference_mode(
                reference["cp_factors"], mode=3, rows=lr_indices
            )
            print(f"   {axis} replicate {replicate + 1}/{n_replicates}")

            weights, factors, r2x, diagnostics = _fit_cp(
                perturbed,
                cp_rank,
                n_iter_max=cp_n_iter_max,
                tol=cp_tol,
                init=init,
                svd=svd,
                random_state=seed,
                compress=compress,
                oversample=oversample,
            )
            weights, factors = standardize_cc_pf2(weights, factors)
            frames.append(
                _tag(
                    per_component_fms(ref_cp, factors, skip_modes=CPD_SKIP_MODES),
                    axis=axis,
                    replicate=replicate,
                    seed=seed,
                    stage=STAGE_CPD,
                    r2x=r2x,
                    diagnostics=diagnostics,
                    n_cells=np.nan,
                    n_conditions=np.nan,
                )
            )
            clean_memory()

    return pd.concat(frames, ignore_index=True)


def _check_rosters(reference: dict[str, Any], fit: dict[str, Any]) -> None:
    """Refuse to compare two fits whose gene, LR-pair or condition modes differ.

    Both rosters are fixed in practice -- gene filtering happens in the loader
    before any resampling, and `ccc.build_context_ccc_tensor` intersects over
    contexts whose gene index is identical -- so this should never fire. It
    raises rather than intersecting because a silent difference is the
    dangerous case: `per_component_fms` would line up different genes or
    different interactions row by row and return a perfectly plausible number,
    and intersecting would change the mode length and so the FMS scale
    invisibly.

    The condition mode is checked by length rather than membership: the
    replicate's roster is *meant* to differ from the reference's on the
    condition axis. What must hold is that `condition_labels` still names
    mode-0 row for row, since that is what `reference_condition_rows` matches
    on. A length mismatch is the loud symptom of the labels having drifted out
    of step with the rows.
    """
    if not np.array_equal(fit["gene_names"], reference["gene_names"]):
        raise ValueError(
            "Replicate and reference fits retained different genes, so their "
            "RISE gene-factor rows are not comparable."
        )
    if not np.array_equal(fit["lr_symbols"], reference["lr_symbols"]):
        raise ValueError(
            "Replicate and reference fits retained different ligand-receptor "
            "pairs, so their CP ligand-receptor factor rows are not "
            "comparable and component matching would be meaningless."
        )
    for fit_or_reference in (fit, reference):
        n_labels = len(fit_or_reference["condition_labels"])
        for key in ("rise_factors", "cp_factors"):
            n_rows = fit_or_reference[key][0].shape[0]
            if n_labels != n_rows:
                raise ValueError(
                    f"{n_labels} condition labels for {n_rows} rows of "
                    f"{key}'s condition mode. The labels are what "
                    "`reference_condition_rows` matches on, so a length "
                    "mismatch means they no longer name the rows they are "
                    "read as naming."
                )


# =============================================================================
# Summaries
# =============================================================================


def metric_interval(
    df: pd.DataFrame,
    metric: str,
    by: tuple[str, ...] = ("Axis", "Stage", "Component"),
    replicate_key: str = "Replicate",
    lower: float = 2.5,
    upper: float = 97.5,
) -> pd.DataFrame:
    """Spread of a metric across replicates, within whatever groups are given.

    A replicate contributes one value per group -- the mean over its rows in
    that group -- so the reported spread is across replicates and never mixes
    in within-replicate variation.

    That averaging step is what makes the same function serve both readings of
    a stability table. Grouped per component, each replicate already has one
    row per group and the mean is an identity, so the result is the
    per-component band. Grouped by ``("Axis", "Stage")`` the mean is taken over
    components within a replicate, which by `per_component_fms`'s construction
    is the aggregate FMS itself -- so the headline number and the per-component
    detail cannot disagree about what a replicate contributed.

    Parameters
    ----------
    df : pd.DataFrame
        Long table carrying `by`, `replicate_key` and `metric`.
    metric : str
        Column to summarize.
    by : tuple[str, ...]
        Grouping keys.
    replicate_key : str, default="Replicate"
        Column identifying a replicate within a group.
    lower, upper : float
        Percentiles reported as the interval.

    Returns
    -------
    pd.DataFrame
        One row per group with the replicate count, mean, median, the
        percentile interval and the observed range.
    """
    per_replicate = df.groupby([*by, replicate_key])[metric].mean().reset_index()
    return (
        per_replicate.groupby(list(by))[metric]
        .agg(
            NReplicates="count",
            Mean="mean",
            Median="median",
            **{
                f"P{lower:g}": lambda x: np.percentile(x, lower),
                f"P{upper:g}": lambda x: np.percentile(x, upper),
            },
            Min="min",
            Max="max",
        )
        .reset_index()
    )


def summarize_component_stability(
    df: pd.DataFrame,
    lower: float = 2.5,
    upper: float = 97.5,
) -> pd.DataFrame:
    """Per-component FMS summary with the seed-refit ceiling beside each component.

    One row per (Axis, Stage, Component): that component's FMS band across
    replicates, its median runner-up score, and -- matched on that same
    component -- the seed axis's median FMS, the lower edge of its
    reproducibility band, and the fraction of replicates in which the
    component stayed inside it.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `component_stability`.
    lower, upper : float
        Percentiles reported as the interval for the observed axes. The seed
        band's own edge is `SEED_BAND_PERCENTILE`, deliberately independent of
        these.

    Returns
    -------
    pd.DataFrame
        The `metric_interval` columns plus 'MedianRunnerUpFMS',
        'SeedMedianFMS', 'SeedP5FMS' and 'FracWithinSeedBand'. The last three
        are NaN when the seed axis was not run: 0.0 would report every
        component as having failed a test that was never performed.
    """
    keys = ["Axis", "Stage", "Component"]
    observed = df[df["Axis"] != SEED_AXIS]
    summary = metric_interval(
        observed, "FMS", by=tuple(keys), lower=lower, upper=upper
    ).rename(columns={"Mean": "MeanFMS", "Median": "MedianFMS"})

    runner_up = (
        observed.groupby(keys)["RunnerUpFMS"]
        .median()
        .rename("MedianRunnerUpFMS")
        .reset_index()
    )
    summary = summary.merge(runner_up, on=keys, how="left")

    ceiling_df = df[df["Axis"] == SEED_AXIS]
    if not ceiling_df.empty:
        ceiling = (
            ceiling_df.groupby(["Stage", "Component"])["FMS"]
            .agg(
                SeedMedianFMS="median",
                SeedP5FMS=lambda x: np.percentile(x, SEED_BAND_PERCENTILE),
            )
            .reset_index()
        )
        summary = summary.merge(ceiling, on=["Stage", "Component"], how="left")
        within = observed.merge(ceiling, on=["Stage", "Component"], how="left")
        within["Within"] = within["FMS"] >= within["SeedP5FMS"]
        frac = (
            within.groupby(keys)
            .agg(
                FracWithinSeedBand=("Within", "mean"),
                NCeiling=("SeedP5FMS", "count"),
                NRows=("FMS", "count"),
            )
            .reset_index()
        )
        # A (Stage, Component) the seed axis never covered would otherwise be
        # scored 0.0 -- indistinguishable from having genuinely failed.
        frac["FracWithinSeedBand"] = frac["FracWithinSeedBand"].where(
            frac["NCeiling"] == frac["NRows"]
        )
        summary = summary.merge(
            frac.drop(columns=["NCeiling", "NRows"]), on=keys, how="left"
        )
    else:
        summary["SeedMedianFMS"] = np.nan
        summary["SeedP5FMS"] = np.nan
        summary["FracWithinSeedBand"] = np.nan

    return summary
