"""
Component and association stability of the CCC-RISE pipeline at fixed RISE and
CP ranks.

This analysis evaluates whether the identified components recovered at those fixed
ranks are stable and reproducible and whether each CP component's association with
the phenotype survives the same perturbations. Two axes perturb the fit:
`JACKKNIFE_AXIS` leaves one patient out per replicate, every patient in turn, and
`SEED_AXIS` refits the identical cohort from a different random seed, the
reproducibility ceiling the jackknife is read against.

One replicate loop serves all three analyses: a replicate prepares the data
once, refits RISE and CP once, scores RISE and CPD component stability from
those fits, and then re-runs the component--phenotype ANOVA on the same CP fit.
Each replicate's ANOVA statistics are carried over onto the reference
component its CP component was matched to by the CPD factor match score, so a
component's association statistics and its FMS always describe the same
replicate component. A component that could not be re-identified (low FMS)
produces a wide association distribution for a reason that has nothing to do
with the phenotype, and keeping both on one row keeps the two explanations
separable.

Distinction between Per-Fit Diagnostics and Cross-Fit Stability Metrics
----------------------------------------------------------------------
This module computes and tracks two distinct categories of metrics:

1. **Per-Fit Diagnostics (Single-Fit Properties)**:
   Diagnostics evaluating the reconstruction quality, identifiability, and
   numerical conditioning of an *individual* model fit on a specific dataset:
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

Matched Modes
-------------
Components are matched, and the FMS computed, only on modes whose coordinates
mean the same thing in the reference and the replicate (`axis_skip_modes`):

- **Eigen-state modes are always held out.** RISE Mode 1 (basis $B$) and CPD
  Modes 1 and 2 (sender $D$ and receiver $E$ eigen-state loadings) are
  expressed in each fit's own RISE basis, which differs from replicate to
  replicate (`RISE_SKIP_MODES`, `CPD_SKIP_MODES`).
- **The condition mode is held out on `JACKKNIFE_AXIS`**, whose roster
  differs from the reference's.

So on `SEED_AXIS` RISE is matched on conditions and genes and CPD on
conditions and LR pairs; on `JACKKNIFE_AXIS` RISE is matched on genes alone
and CPD on LR pairs alone. FMS values therefore rest on different modes on the
two axes.
"""

from pathlib import Path
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

from ..anova import anova_component_associations, prepare_anova_inputs
from ..ccc_rise import (
    add_cell_embeddings,
    run_ccc_rise_workflow,
    save_ccc_rise_results,
)
from ..factors import clean_memory, load_factors_netcdf
from ..utils import (
    jackknife_units,
    leave_out_patient,
    replicate_seed,
)

Factors = list[np.ndarray]

#: Values of the 'Stage' column.
STAGE_RISE = "RISE"
STAGE_CPD = "CPD"

#: Eigen-state modes, held out of every factor match score (FMS): their
#: coordinates are each fit's own RISE basis.
RISE_SKIP_MODES: tuple[int, ...] = (1,)
CPD_SKIP_MODES: tuple[int, ...] = (1, 2)
#: The condition mode of both stages.
CONDITION_MODE = 0

#: Modes `align_to_reference_components` matches two CP fits over by default:
#: the ligand-receptor mode alone, the one mode whose coordinates mean the same
#: thing in two fits with separately-fit RISE factors and different rosters.
LR_MATCH_MODES: tuple[int, ...] = (3,)

#: Per-component ANOVA columns carried onto a replicate's CPD rows.
ANOVA_COLUMNS = ("EtaSquared", "F", "P", "PAdj", "GroupMeanDiff")

#: Name of the reproducibility-ceiling axis: identical data, a different random
#: seed, refit end to end. `random_state` genuinely moves both stages: `parafac2_nd`
#: seeds the Gaussian projection of its randomized-SVD initialization and its
#: compression stage, and tensorly's ``parafac(init="svd", svd="randomized_svd")`` does
#: the same, so the axis is not degenerate.
SEED_AXIS = "seed_only"

#: Percentile of the seed axis's FMS taken as the reproducibility band's lower
#: edge.
SEED_BAND_PERCENTILE = 5.0

#: Name of the leave-one-patient-out axis: one replicate per patient, each
#: refitting the pipeline on the cohort without that patient
#: (`utils.leave_out_patient`). It runs one replicate per patient
#: (`utils.jackknife_units`) regardless of `n_replicates`, and holds out the
#: condition mode, whose roster it changes, from its FMS.
JACKKNIFE_AXIS = "jackknife"

#: Axes run by default -- every axis there is.
DEFAULT_AXES = (SEED_AXIS, JACKKNIFE_AXIS)


# =============================================================================
# Matching two fits' components
# =============================================================================


def axis_skip_modes(stage: str, axis: str) -> tuple[int, ...]:
    """Modes held out of `stage`'s FMS on `axis`: the stage's eigen-state
    modes, plus the condition mode on `JACKKNIFE_AXIS` (see the module
    docstring)."""
    skip = RISE_SKIP_MODES if stage == STAGE_RISE else CPD_SKIP_MODES
    if axis == JACKKNIFE_AXIS:
        skip += (CONDITION_MODE,)
    return tuple(sorted(skip))


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
    ratio whose denominator can approach zero, so the index itself moves
    between fits even when the components do not.

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


def _signed_mode_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed cosine similarity between every column of `a` and every column of `b`."""
    a_norm = a / np.linalg.norm(a, axis=0, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=0, keepdims=True)
    return a_norm.T @ b_norm


def align_to_reference_components(
    reference_factors: Factors,
    factors: Factors,
    values: np.ndarray,
    modes: tuple[int, ...] = LR_MATCH_MODES,
) -> pd.DataFrame:
    """Relabel one CP fit's per-component values by the reference fit's
    component numbering, correcting for sign.

    CP fixes neither component order nor component sign across separate fits,
    so a per-component quantity is not comparable between two fits until each
    of the second fit's components has been identified with a reference
    component and re-expressed in that component's orientation. Optimal
    assignment gives the relabeling, and the matched mode's *signed* cosine
    gives the sign, so a value carried by a gauge-flipped component is negated
    rather than being recorded as a genuine change of direction. Matching on
    several modes scores a pair by its mean absolute cosine across them and
    leaves values uncorrected for sign, since each mode can carry its own gauge
    flip.

    The replicate loop does not use this -- it matches by the CPD FMS (see
    `_attach_anova`); it relabels per-component values between communication
    scores' fits (`figures.figure3.figureS_ccc_scoring`).

    Parameters
    ----------
    reference_factors : list[np.ndarray]
        The reference fit's CP factors [A, B, C, D]; its component numbering
        is the one every aligned fit is expressed in.
    factors : list[np.ndarray]
        The fit being aligned, sharing `reference_factors`' dimensions along
        every mode in `modes`.
    values : np.ndarray
        One value per component of `factors`, in that fit's own order.
    modes : tuple[int, ...], default=LR_MATCH_MODES
        Modes the match is computed over; defaults to the ligand-receptor
        mode alone.

    Returns
    -------
    pd.DataFrame
        One row per reference component matched (1-indexed), columns
        'Component', 'MatchedComponent', 'Value' (sign-corrected),
        'MatchScore', 'SignedCosine', 'RunnerUpScore'.
    """
    signed_cos_by_mode = {
        mode: _signed_mode_cosine(reference_factors[mode], factors[mode])
        for mode in modes
    }
    combined = np.mean([np.abs(cos) for cos in signed_cos_by_mode.values()], axis=0)

    row_ind, col_ind = linear_sum_assignment(-combined)

    rows = []
    for i, j in zip(row_ind, col_ind, strict=True):
        if combined.shape[1] > 1:
            alt_scores = combined[i].copy()
            alt_scores[j] = -np.inf
            runner_up_j = int(np.argmax(alt_scores))
            runner_up_score: float = float(alt_scores[runner_up_j])
        else:
            runner_up_score = np.nan

        signed_col = modes[0] if len(modes) == 1 else None
        signed_cos = (
            signed_cos_by_mode[signed_col][i, j] if signed_col is not None else np.nan
        )
        sign_flip = np.sign(signed_cos) if signed_col is not None else 1.0

        rows.append(
            {
                "Component": i + 1,
                "MatchedComponent": j + 1,
                "Value": float(values[j] * sign_flip),
                "MatchScore": float(combined[i, j]),
                "SignedCosine": float(signed_cos),
                "RunnerUpScore": runner_up_score,
            }
        )

    return pd.DataFrame(rows).sort_values("Component").reset_index(drop=True)


def top_lr_pairs(
    lr_factor: np.ndarray, lr_symbols: np.ndarray, top_n: int = 3
) -> pd.DataFrame:
    """The ligand-receptor pairs each component of one CP fit loads onto most
    strongly -- a human-readable label for what a component "is", without
    hand-picking an example pair up front.

    Parameters
    ----------
    lr_factor : np.ndarray
        The CP fit's LR-pair factor matrix (D), shape (n_lr_pairs, rank).
    lr_symbols : np.ndarray
        LR pair symbols matching `lr_factor`'s row order.
    top_n : int, default=3
        Number of pairs, by absolute loading, to report per component.

    Returns
    -------
    pd.DataFrame
        One row per (component, top-N slot), columns 'Component', 'LRPair',
        'DLoading' (signed), 'TopRank' (1 = largest magnitude).
    """
    lr_symbols = np.asarray(lr_symbols)
    rows = []
    for component in range(1, lr_factor.shape[1] + 1):
        loadings = lr_factor[:, component - 1]
        top_idx = np.argsort(np.abs(loadings))[::-1][:top_n]
        rows.extend(
            {
                "Component": component,
                "LRPair": str(lr_symbols[idx]),
                "DLoading": float(loadings[idx]),
                "TopRank": slot,
            }
            for slot, idx in enumerate(top_idx, start=1)
        )
    return pd.DataFrame(rows)


# =============================================================================
# Component--phenotype association
# =============================================================================


def anova_data(
    adata: anndata.AnnData, condition_column: str, group_column: str
) -> anndata.AnnData:
    """The slice of a fitted AnnData the component ANOVA reads.

    `anova.prepare_anova_inputs` needs only each cell's condition index and
    group label and its total counts (for `utils.correct_conditions`' read
    depth correction), so this keeps those as a one-column AnnData that a
    fit can carry cheaply.
    """
    totals = np.asarray(cast(Any, adata.X).sum(axis=1), dtype=float).reshape(-1, 1)
    obs = cast(pd.DataFrame, adata.obs)
    return anndata.AnnData(
        X=totals,
        obs=obs[[condition_column, group_column, "condition_unique_idxs"]].copy(),
    )


def component_anova(
    data: anndata.AnnData,
    cp_factors: Factors,
    condition_column: str,
    group_column: str,
    control_label: str | None = None,
    other_label: str | None = None,
) -> pd.DataFrame:
    """One-way ANOVA of each CP component's condition loading against
    `group_column`; see `anova.anova_component_associations`.

    A group left with a single condition by a replicate (e.g. ALAD's
    two-condition 'Died' outcome when the jackknife leaves one)
    stays in the test as a group of one rather than voiding it, so every
    replicate with at least two groups gets statistics.

    Parameters
    ----------
    data : anndata.AnnData
        `anova_data` of the cohort the CP fit's condition mode indexes.
    cp_factors : list[np.ndarray]
        CP factors [A, B, C, D].
    condition_column, group_column : str
        Columns in `obs` holding the condition identity and its group label.
    control_label, other_label : str | None, optional
        Values of `group_column` collapsed to the reference and other level.
        Both None -- the default -- tests every level of `group_column`; see
        `anova.prepare_anova_inputs`.
    """
    X, sample_to_group = prepare_anova_inputs(
        data, cp_factors, condition_column, group_column, control_label, other_label
    )
    return anova_component_associations(
        X,
        sample_to_group,
        control_label=control_label,
        correction="fdr_bh",
        min_group_size=1,
    )


def _attach_anova(block: pd.DataFrame, anova: pd.DataFrame | None) -> pd.DataFrame:
    """Carry a replicate's ANOVA statistics onto the CPD rows of
    `per_component_fms`, by the reference component each was matched to.

    F, P, PAdj and eta-squared do not change under a gauge flip and are read straight
    off the matched replicate component. `GroupMeanDiff` is a difference of
    condition-mode loadings, so it is sign-corrected by the condition mode's
    own signed cosine (`SignedCosMode0`) where that mode was compared, and by
    the ligand-receptor mode's (`SignedCosMode3`) where it was held out.
    """
    if anova is None:
        return block
    matched = block["MatchedComponent"].to_numpy(dtype=int) - 1
    for column in ANOVA_COLUMNS:
        block[column] = anova[column].to_numpy(dtype=float)[matched]
    sign_cosine = block["SignedCosMode0"].fillna(block["SignedCosMode3"])
    block["GroupMeanDiff"] *= np.sign(sign_cosine.to_numpy())
    return block


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
    Consumers -- the core-numbering check in `relabel_rise` above all -- treat
    this array as positionally aligned to mode 0, and a silent disagreement
    would compare the wrong patients and still return a plausible number.
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
    control_label: str | None = None,
    other_label: str | None = None,
    complex_sep: str | None = "&",
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
    random_state: int = 42,
    backend: str | None = None,
    compress: bool = False,
    numbering_reference: xr.Dataset | Path | None = None,
    *,
    rise_n_iter_max: int,
    cp_n_iter_max: int,
    rise_tol: float,
    cp_tol: float,
) -> dict[str, Any]:
    """Run RISE, build the communication tensor, and fit CP over `adata`.

    `adata` is written into by `parafac2.store_pf2`, so callers must hand
    over a copy they do not need pristine -- which is what
    `stability_replicates` hands it.

    Parameters
    ----------
    adata : anndata.AnnData
        Data to fit: the full cohort for the reference pass, a replicate's
        copy otherwise.
    lr_pairs : pd.DataFrame
        Ligand-receptor pairs to score.
    condition_column : str
        Column in `obs` holding the condition (patient) identity.
    rise_rank, cp_rank : int
        Ranks for the RISE and CP stages.
    group_column : str | None, default=None
        Condition-level phenotype column the CP components are tested against
        (`component_anova`); no ANOVA is run when None.
    control_label, other_label : str | None, optional
        Values of `group_column` collapsed to the reference and other level;
        both None tests every level. See `anova.prepare_anova_inputs`.
    rise_n_iter_max, rise_tol : int, float
        Fitting controls for RISE.
    cp_n_iter_max, cp_tol
        Fitting controls for CP.
    complex_sep, communication_score, hill_k
        Communication-tensor options; see `ccc_rise.calc_communication_score`.
    random_state : int, default=42
        Seed for the RISE fit and the CP fit. The ANOVA is deterministic.
    backend : str | None, default=None
        Backend the RISE fit runs on ('cpu', 'cupy', 'mlx', or None to
        auto-detect); passed straight through to `ccc_rise.fit_rise`.
    compress : bool, default=False
        Fit RISE and CP compressed; see `ccc_rise.run_ccc_rise_workflow`.
    numbering_reference : xr.Dataset | Path | None, optional
        Core factors fit to relabel components into (`relabel_workflow_fit`);
        the reference pass sets it, replicates are matched to that reference.

    Returns
    -------
    dict[str, Any]
        Fitted factors, diagnostics, metadata, and the ANOVA; see `stability_fit`.
    """
    fit = run_ccc_rise_workflow(
        adata,
        rise_rank,
        lr_pairs,
        cp_rank=cp_rank,
        condition_column=condition_column,
        rise_n_iter_max=rise_n_iter_max,
        rise_tol=rise_tol,
        cp_n_iter_max=cp_n_iter_max,
        cp_tol=cp_tol,
        complex_sep=complex_sep,
        communication_score=communication_score,
        hill_k=hill_k,
        random_state=random_state,
        backend=backend,
        compress=compress,
    )
    if numbering_reference is not None:
        fit = relabel_workflow_fit(fit, numbering_reference, condition_column)
    return stability_fit(
        fit["adata"], fit, condition_column, group_column, control_label, other_label
    )


def stability_fit(
    adata: anndata.AnnData,
    fit: dict[str, Any],
    condition_column: str,
    group_column: str | None = None,
    control_label: str | None = None,
    other_label: str | None = None,
) -> dict[str, Any]:
    """`fit_stability_pipeline`'s result, from a RISE-fit `adata` and a
    `ccc_rise.fit_ccc_cp` / `run_ccc_rise_workflow` result. With
    `group_column`, also carries the fit's `anova_data` and its per-component
    `component_anova` table ('anova_data' and 'anova' are None otherwise).
    """
    data = anova = None
    if group_column is not None:
        data = anova_data(adata, condition_column, group_column)
        anova = component_anova(
            data,
            fit["cp_factors"],
            condition_column,
            group_column,
            control_label,
            other_label,
        )
    return {
        "rise_factors": [
            adata.uns["Pf2_A"],
            adata.uns["Pf2_B"],
            np.asarray(adata.varm["Pf2_C"]),
        ],
        "cp_weights": fit["cp_weights"],
        "cp_factors": fit["cp_factors"],
        "condition_labels": _condition_labels(adata, condition_column),
        "gene_names": np.asarray(adata.var_names),
        "lr_symbols": fit["filtered_lr_pairs"]["interaction_symbol"].to_numpy(),
        "r2x_rise": float(adata.uns["r2x_pf2"]),
        "r2x_cp": fit["r2x_cp"],
        "cpd_diagnostics": fit["diagnostics"],
        "anova_data": data,
        "anova": anova,
    }


# =============================================================================
# Core-fit numbering
# =============================================================================
#
# `ccc_rise.standardize_cc_pf2` orders components by a scale-dependent ratio,
# so two fits of the same data (the core `cccrise-run` fit and a validation
# runner's compressed reference, say) can number the same components
# differently. Validation runners relabel their reference fit into the core
# fit's numbering here, matching components by FMS (RISE on the condition and
# gene modes, CP on the condition and ligand-receptor modes). Relabeling is a
# pure permutation; the fits themselves are unchanged.


def relabel_workflow_fit(
    fit: dict[str, Any], core: xr.Dataset | Path, condition_column: str
) -> dict[str, Any]:
    """Relabel a `ccc_rise.run_ccc_rise_workflow` result, in place, into the
    core fit's RISE and CP numbering, and return it."""
    core = _as_dataset(core)
    adata = fit["adata"]
    rise_order = relabel_rise(adata, core, condition_column)
    relabel_cp_fit(fit, core, rise_order)
    save_ccc_rise_results(
        adata, fit["cp_factors"], fit["cp_weights"], fit["filtered_lr_pairs"]
    )
    add_cell_embeddings(adata, fit["cp_factors"][1], fit["cp_factors"][2], False, None)
    return fit


def relabel_rise(
    adata: anndata.AnnData, core: xr.Dataset | Path, condition_column: str
) -> np.ndarray:
    """Permute a `ccc_rise.fit_rise` AnnData's RISE components, in place, into
    the core fit's numbering.

    Returns the order applied: ``order[i]`` is the (0-based) component of this
    fit now numbered ``i``. `Pf2_B` is the identity under `fix_B_identity`, so
    projection columns are components and permute with them.
    """
    core = _as_dataset(core)
    _check_core_axis(core, "condition", _condition_labels(adata, condition_column))
    _check_core_axis(core, "gene", adata.var_names)
    order = _core_match_order(
        [core["pf2_A"].values, core["pf2_B"].values, core["pf2_C"].values],
        [adata.uns["Pf2_A"], adata.uns["Pf2_B"], np.asarray(adata.varm["Pf2_C"])],
        RISE_SKIP_MODES,
    )
    adata.uns["Pf2_A"] = adata.uns["Pf2_A"][:, order]
    adata.uns["Pf2_B"] = adata.uns["Pf2_B"][np.ix_(order, order)]
    adata.uns["Pf2_weights"] = adata.uns["Pf2_weights"][order]
    adata.varm["Pf2_C"] = np.asarray(adata.varm["Pf2_C"])[:, order]
    for key in ("projections", "weighted_projections"):
        if key in adata.obsm:
            adata.obsm[key] = np.asarray(adata.obsm[key])[:, order]
    return order


def relabel_cp_fit(
    fit: dict[str, Any], core: xr.Dataset | Path, rise_order: np.ndarray | None
) -> np.ndarray:
    """Permute a `ccc_rise.fit_ccc_cp` result's CP components, in place, into
    the core fit's numbering; returns the order applied.

    `rise_order` is a RISE relabeling applied to the fit's AnnData *after* its
    tensor was built (as in `relabel_workflow_fit`); the tensor's and B/C's
    eigen-state axes are permuted to follow it. Pass None when the tensor was
    built from an already-relabeled RISE fit.
    """
    core = _as_dataset(core)
    lr_symbols = fit["filtered_lr_pairs"]["interaction_symbol"].to_numpy()
    _check_core_axis(core, "lr_pair", lr_symbols)
    factors = list(fit["cp_factors"])
    if rise_order is not None:
        factors[1], factors[2] = factors[1][rise_order], factors[2][rise_order]
        if fit.get("tensor") is not None:
            fit["tensor"] = fit["tensor"][:, rise_order][:, :, rise_order]
    order = _core_match_order(
        [core[f"cp_{mode}"].values for mode in "ABCD"], factors, CPD_SKIP_MODES
    )
    fit["cp_factors"] = [f[:, order] for f in factors]
    fit["cp_weights"] = fit["cp_weights"][order]
    return order


def _core_match_order(
    core_factors: Factors, factors: Factors, skip_modes: tuple[int, ...]
) -> np.ndarray:
    if core_factors[0].shape[1] != factors[0].shape[1]:
        raise ValueError(
            f"Core fit has rank {core_factors[0].shape[1]} but this fit has rank "
            f"{factors[0].shape[1]}; numbering can only be shared at equal ranks."
        )
    matched = per_component_fms(core_factors, factors, skip_modes=skip_modes)
    return matched["MatchedComponent"].to_numpy() - 1


def _check_core_axis(core: xr.Dataset, dim: str, labels) -> None:
    if not np.array_equal(core[dim].values.astype(str), np.asarray(labels).astype(str)):
        raise ValueError(
            f"This fit's {dim!r} axis does not match the core fit's, so its "
            "components cannot be matched to the core numbering."
        )


def _as_dataset(core: xr.Dataset | Path) -> xr.Dataset:
    return core if isinstance(core, xr.Dataset) else load_factors_netcdf(core)


# =============================================================================
# The shared replicate loop
# =============================================================================


def stability_replicates(
    adata: anndata.AnnData,
    reference: dict[str, Any],
    lr_pairs: pd.DataFrame,
    condition_column: str,
    rise_rank: int,
    cp_rank: int,
    group_column: str | None = None,
    control_label: str | None = None,
    other_label: str | None = None,
    patient_column: str | None = None,
    axes: tuple[str, ...] = DEFAULT_AXES,
    n_replicates: int = 20,
    complex_sep: str | None = "&",
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
    random_state: int = 42,
    backend: str | None = None,
    compress: bool = False,
    *,
    rise_n_iter_max: int,
    cp_n_iter_max: int,
    rise_tol: float,
    cp_tol: float,
) -> pd.DataFrame:
    """Per-component stability of both stages, and of the CP components'
    phenotype associations, across replicate refits, in long form.

    Every replicate prepares the data, refits RISE at `rise_rank`, and matches
    the resulting components onto the reference's; the communication tensor
    built from that same fit is then decomposed at `cp_rank` and matched in
    turn, and the same CP fit's ANOVA against `group_column` is carried onto
    its matched components (`_attach_anova`). Every match holds out the
    eigen-state modes, and on `JACKKNIFE_AXIS` the condition mode
    (`axis_skip_modes`). `JACKKNIFE_AXIS` leaves one patient out per
    replicate, every patient in turn, so each of its replicates measures that
    one patient's influence on every component and association. `SEED_AXIS`
    hands the pipeline the identical cohort and changes only the seed the fits
    start from, so it measures how far components move under the fitters' own
    randomness -- the reproducibility ceiling the jackknife is read against.

    Parameters
    ----------
    adata : anndata.AnnData
        Full-cohort data.
    reference : dict[str, Any]
        The full-cohort fit, produced by `fit_stability_pipeline` with the
        same `group_column`, `control_label` and `other_label`.
    lr_pairs : pd.DataFrame
        Ligand-receptor pairs to score.
    condition_column : str
        Column in `obs` holding the condition (patient) identity.
    rise_rank, cp_rank : int
        Ranks for the RISE and CP stages.
    group_column : str | None, default=None
        Condition-level phenotype column the CP components are tested against;
        no ANOVA is run when None.
    control_label, other_label : str | None, optional
        Values of `group_column` collapsed to the reference and other level;
        both None tests every level. See `anova.prepare_anova_inputs`.
    patient_column : str | None, default=None
        Condition-level patient identity column: the unit `JACKKNIFE_AXIS`
        leaves out, so a patient contributing several conditions loses all of
        them (`condition_column` when None).
    axes : tuple[str, ...], default=DEFAULT_AXES
        Which axes to run.
    n_replicates : int, default=20
        Replicates on `SEED_AXIS`; `JACKKNIFE_AXIS` always runs one per
        patient.
    rise_n_iter_max, rise_tol : int, float
        Fitting controls for RISE.
    cp_n_iter_max, cp_tol
        Fitting controls for CP.
    complex_sep, communication_score, hill_k
        Communication-tensor options.
    random_state : int, default=42
        Base seed; replicate ``r`` of every axis seeds its fits with
        `replicate_seed(random_state, r)`.
    backend : str | None, default=None
        Backend each replicate's RISE refit runs on; see
        `fit_stability_pipeline`.
    compress : bool, default=False
        Fit RISE and CP compressed; see `ccc_rise.run_ccc_rise_workflow`.

    Returns
    -------
    pd.DataFrame
        Long-form table with one row per (axis, replicate, stage, reference
        component): the `per_component_fms` match, per-fit diagnostics, the
        patient a `JACKKNIFE_AXIS` replicate left out ('LeftOut', None on
        `SEED_AXIS`), and -- on CPD rows, when `group_column` is given -- the
        matched component's `ANOVA_COLUMNS`.
    """
    unknown = set(axes) - set(DEFAULT_AXES)
    if unknown:
        raise ValueError(f"Unknown stability axes: {sorted(unknown)}")

    fit_kwargs: dict[str, Any] = {
        "lr_pairs": lr_pairs,
        "condition_column": condition_column,
        "group_column": group_column,
        "control_label": control_label,
        "other_label": other_label,
        "rise_rank": rise_rank,
        "cp_rank": cp_rank,
        "rise_n_iter_max": rise_n_iter_max,
        "rise_tol": rise_tol,
        "cp_n_iter_max": cp_n_iter_max,
        "cp_tol": cp_tol,
        "complex_sep": complex_sep,
        "communication_score": communication_score,
        "hill_k": hill_k,
        "backend": backend,
        "compress": compress,
    }
    unit_column = patient_column or condition_column
    patients = jackknife_units(adata, unit_column)

    frames = []
    for axis in axes:
        n_axis = len(patients) if axis == JACKKNIFE_AXIS else n_replicates
        for replicate in range(n_axis):
            seed = replicate_seed(random_state, replicate)
            print(f"   {axis} replicate {replicate + 1}/{n_axis}")
            left_out = str(patients[replicate]) if axis == JACKKNIFE_AXIS else None
            data = (
                adata.copy()
                if left_out is None
                else leave_out_patient(adata, unit_column, left_out)
            )
            fit = fit_stability_pipeline(data, random_state=seed, **fit_kwargs)
            block = _replicate_block(reference, fit, axis)
            block.insert(0, "Seed", seed)
            block.insert(0, "Replicate", replicate)
            block.insert(0, "Axis", axis)
            block["LeftOut"] = left_out
            block["NCells"] = float(data.n_obs)
            block["NConditions"] = float(len(fit["condition_labels"]))
            frames.append(block)
            # CuPy's pool never returns freed blocks to the driver on its own,
            # so a long sweep of variably-sized fits fragments it until a fit
            # spuriously trips the managed-memory fallback.
            clean_memory()

    return pd.concat(frames, ignore_index=True)


def _stage_block(
    reference_factors: Factors,
    factors: Factors,
    stage: str,
    axis: str,
    r2x: float,
    diagnostics: dict[str, Any] | None = None,
    anova: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One fit's `per_component_fms` rows for `stage`, with its per-fit
    diagnostics and, on CPD rows, its matched ANOVA statistics."""
    block = per_component_fms(
        reference_factors, factors, skip_modes=axis_skip_modes(stage, axis)
    )
    block.insert(0, "Stage", stage)
    block["R2X"] = r2x
    for column in ("CoreConsistency", "MaxModesCollinear", "WorstPairCanceling"):
        block[column] = (diagnostics or {}).get(column, np.nan)
    return _attach_anova(block, anova)


def _replicate_block(
    reference: dict[str, Any], fit: dict[str, Any], axis: str
) -> pd.DataFrame:
    """RISE and CPD rows of one replicate's refit."""
    _check_rosters(reference, fit, axis)
    rise = _stage_block(
        reference["rise_factors"],
        fit["rise_factors"],
        STAGE_RISE,
        axis,
        fit["r2x_rise"],
    )
    cpd = _stage_block(
        reference["cp_factors"],
        fit["cp_factors"],
        STAGE_CPD,
        axis,
        fit["r2x_cp"],
        fit["cpd_diagnostics"],
        fit["anova"],
    )
    return pd.concat([rise, cpd], ignore_index=True)


def _check_rosters(reference: dict[str, Any], fit: dict[str, Any], axis: str) -> None:
    """Refuse to compare two fits on a mode whose rows differ between them.

    The gene and LR-pair rosters are fixed in practice -- gene filtering happens
    in the loader before any replicate is drawn, and `ccc.build_context_ccc_tensor`
    intersects over contexts whose gene index is identical -- so this should
    never fire. It raises rather than intersecting because a silent difference
    is the dangerous case: `per_component_fms` would line up different genes or
    different interactions row by row and return a perfectly plausible number.
    The condition roster must match too, on every axis that compares the
    condition mode (all but `JACKKNIFE_AXIS`).
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
    if axis != JACKKNIFE_AXIS and not np.array_equal(
        fit["condition_labels"], reference["condition_labels"]
    ):
        raise ValueError(
            f"The {axis!r} axis compares the condition mode, but the replicate's "
            "conditions differ from the reference's."
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
        Output of `stability_replicates`.
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


def summarize_anova_stability(
    df: pd.DataFrame,
    reference_anova: pd.DataFrame,
    axis: str = JACKKNIFE_AXIS,
    ci_level: float = 95.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One row per CP component: its effect-size distribution on one axis, and
    how reliably the association and its direction survive the replicates.

    Three things are reported side by side rather than collapsed into one
    number, because they call for different conclusions: how large the effect
    is and how tightly the replicates pin it down (the eta-squared interval);
    how often the association reaches `alpha` at all, by its p-value
    Benjamini-Hochberg corrected across the replicate's components
    (`FracSignificant`); and
    whether the component was even re-identifiable across replicates
    (`MedianFMS`/`MinFMS`), which is a CP identifiability question rather than
    a question about the phenotype.

    There is deliberately no "interval excludes zero" column: eta-squared is a
    variance ratio and cannot be negative, so such a test would be vacuous.
    `FracSignificant` takes its place -- read as a descriptive tally.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `stability_replicates`, run with a `group_column`.
    reference_anova : pd.DataFrame
        The full-cohort fit's per-component ANOVA table -- `stability_fit`'s
        ``result["anova"]`` -- carrying 'Component', 'EtaSquared', 'F', 'P',
        'PAdj' and 'GroupMeanDiff'.
    axis : str, default=JACKKNIFE_AXIS
        Axis to summarize.
    ci_level : float, default=95.0
        Central interval width, in percent, taken on eta-squared: on
        `JACKKNIFE_AXIS` the normal interval around the reference value with
        the jackknife standard error, on `SEED_AXIS` the replicates' percentile
        interval (`_eta_squared_interval`).
    alpha : float, default=0.05
        Threshold `FracSignificant` counts replicates against.

    Returns
    -------
    pd.DataFrame
        One row per component, sorted by component, with columns 'Component',
        'ReferenceEtaSquared', 'ReferenceF', 'ReferenceP', 'ReferencePAdj',
        'MeanEtaSquared',
        'MedianEtaSquared', 'SDEtaSquared', 'CILow', 'CIHigh', 'MedianP',
        'FracSignificant', 'ReferenceGroupMeanDiff', 'MedianGroupMeanDiff',
        'FracSameSignAsReference' (NaN when the reference group-mean
        difference is exactly zero, i.e. has no direction to agree with),
        'MedianFMS', 'MinFMS', 'NReplicates'.
    """
    on_axis = df[(df["Axis"] == axis) & (df["Stage"] == STAGE_CPD)]
    reference_by_component = reference_anova.set_index("Component")
    rows = []
    for component_key, group in on_axis.groupby("Component"):
        component = cast(int, component_key)
        eta = group["EtaSquared"].to_numpy(dtype=float)
        p_values = group["P"].to_numpy(dtype=float)
        q_values = group["PAdj"].to_numpy(dtype=float)
        diffs = group["GroupMeanDiff"].to_numpy(dtype=float)
        reference = reference_by_component.loc[component]
        reference_diff = float(reference["GroupMeanDiff"])
        ci_low, ci_high = _eta_squared_interval(
            eta, float(reference["EtaSquared"]), axis == JACKKNIFE_AXIS, ci_level
        )
        rows.append(
            {
                "Component": component,
                "ReferenceEtaSquared": float(reference["EtaSquared"]),
                "ReferenceF": float(reference["F"]),
                "ReferenceP": float(reference["P"]),
                "ReferencePAdj": float(reference["PAdj"]),
                "MeanEtaSquared": float(np.nanmean(eta)),
                "MedianEtaSquared": float(np.nanmedian(eta)),
                "SDEtaSquared": float(np.nanstd(eta, ddof=1)) if len(eta) > 1 else 0.0,
                "CILow": float(ci_low),
                "CIHigh": float(ci_high),
                "MedianP": float(np.nanmedian(p_values)),
                # A replicate whose ANOVA came back degenerate (NaN p) counts
                # as not significant rather than dropping out of the tally.
                "FracSignificant": float(np.mean(q_values < alpha)),
                "ReferenceGroupMeanDiff": reference_diff,
                "MedianGroupMeanDiff": float(np.nanmedian(diffs)),
                "FracSameSignAsReference": (
                    float(np.mean(np.sign(diffs) == np.sign(reference_diff)))
                    if np.isfinite(reference_diff) and reference_diff != 0
                    else np.nan
                ),
                "MedianFMS": float(np.median(group["FMS"])),
                "MinFMS": float(np.min(group["FMS"])),
                "NReplicates": len(eta),
            }
        )
    return pd.DataFrame(rows).sort_values("Component").reset_index(drop=True)


def _eta_squared_interval(
    eta: np.ndarray, reference_eta: float, jackknife: bool, ci_level: float
) -> tuple[float, float]:
    """Central `ci_level` interval on eta-squared from one axis's replicates.

    Leave-one-out estimates sit far closer together than independent samples
    would, so their percentiles would understate the uncertainty about
    (n - 1)-fold in variance; the jackknife instead scales their spread up to
    a standard error, ``sqrt((n - 1) / n * sum((eta_i - mean)^2))``, and the
    interval is the normal one around the full-cohort value, floored at 0.
    The seed axis's replicates are refits of one cohort, so their percentile
    interval is the fitters' own spread. Degenerate (NaN) replicates are
    dropped either way.
    """
    eta = eta[np.isfinite(eta)]
    if eta.size == 0:
        return np.nan, np.nan
    tail = (100.0 - ci_level) / 2.0
    if not jackknife:
        low, high = np.percentile(eta, [tail, 100.0 - tail])
        return float(low), float(high)
    n = eta.size
    se = np.sqrt((n - 1) / n * np.sum((eta - eta.mean()) ** 2))
    half_width = norm.ppf(1.0 - tail / 100.0) * se
    return max(reference_eta - half_width, 0.0), reference_eta + half_width
