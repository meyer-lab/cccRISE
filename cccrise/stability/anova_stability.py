"""
Patient-level bootstrap of the CCC-RISE component--phenotype ANOVA.

A single one-way ANOVA on one CCC-RISE factorization reports an F, a p and an
effect size per CP component, but says nothing about how much of that is
attributable to *which patients happened to be enrolled*. Resampling the
patient roster with replacement and re-running the whole pipeline -- RISE, the
interaction tensor, CP at the same fixed ranks, then the ANOVA -- turns each
component's effect size into a distribution whose spread is exactly that
sampling variability.

Every replicate refits RISE and CP from scratch, so its components come out in
an arbitrary order and sign; each replicate is therefore matched back onto the
full-cohort ("reference") fit's components before its statistics are recorded
(`align_to_reference_components`). The match is computed on the
ligand-receptor mode alone, the one mode whose coordinates mean the same thing
in two fits with separately-fit RISE factors, and its score is kept alongside
every row: a component that could not be re-identified in a replicate produces
a wide distribution for a reason that has nothing to do with the phenotype,
and the two explanations have to stay separable. Only `GroupMeanDiff` carries
a sign and so is the only column the alignment's sign correction touches; F, P
and eta-squared are sign-invariant.
"""

from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..anova import anova_component_associations, prepare_anova_inputs
from ..ccc_rise import standardize_cc_pf2, validate_cp_rank
from ..import_data import add_cond_idxs
from ..tensor import calculate_interaction_tensor, rise_store_r2x
from ..utils import replicate_seed, resample_conditions
from .component_stability import _fit_cp

Factors = list[np.ndarray]

#: Modes `align_to_reference_components` matches a replicate to the reference
#: over. Only the ligand-receptor mode (3) survives a separate RISE fit with
#: its coordinates intact: the condition mode indexes a different patient
#: roster, and the sender/receiver modes are expressed in that replicate's own
#: RISE eigen-states.
BOOTSTRAP_MATCH_MODE_WEIGHTS: dict[int, float] = {3: 1.0}


def _signed_mode_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed cosine similarity between every column of `a` and every column of `b`."""
    a_norm = a / np.linalg.norm(a, axis=0, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=0, keepdims=True)
    return a_norm.T @ b_norm


def align_to_reference_components(
    reference_factors: Factors,
    factors: Factors,
    values: np.ndarray,
    mode_weights: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Relabel one CP fit's per-component values by the reference fit's
    component numbering, correcting for sign.

    CP fixes neither component order nor component sign across separate fits,
    so a per-component quantity (a logistic regression weight, say) is not
    comparable between two fits until each of the second fit's components has
    been identified with a reference component and re-expressed in that
    component's orientation. Optimal assignment gives the relabeling, and
    the matched mode's *signed* cosine gives the sign, so a value carried by a
    gauge-flipped component is negated rather than being recorded as a genuine
    change of direction.

    Parameters
    ----------
    reference_factors : list[np.ndarray]
        The reference fit's CP factors [A, B, C, D]; its component numbering
        is the one every aligned fit is expressed in.
    factors : list[np.ndarray]
        The fit being aligned, sharing `reference_factors`' dimensions along
        every mode in `mode_weights`.
    values : np.ndarray
        One value per component of `factors`, in that fit's own order.
    mode_weights : dict[int, float] | None, optional
        Modes the match is computed over. Defaults to `{3: 1.0}` (the
        ligand-receptor mode).

    Returns
    -------
    pd.DataFrame
        One row per reference component matched (1-indexed), columns
        'Component', 'MatchedComponent', 'Value' (sign-corrected),
        'MatchScore', 'SignedCosine', 'RunnerUpScore'.
    """
    if mode_weights is None:
        mode_weights = BOOTSTRAP_MATCH_MODE_WEIGHTS

    signed_cos_by_mode = {
        mode: _signed_mode_cosine(reference_factors[mode], factors[mode])
        for mode in mode_weights
    }

    weight_sum = sum(mode_weights.values())
    combined: np.ndarray = (
        sum(
            (w * np.abs(signed_cos_by_mode[mode]) for mode, w in mode_weights.items()),
            start=np.zeros_like(next(iter(signed_cos_by_mode.values()))),
        )
        / weight_sum
    )

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

        signed_col = next(iter(mode_weights)) if len(mode_weights) == 1 else None
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


def fit_anova_pipeline(
    adata: anndata.AnnData,
    condition_column: str,
    group_column: str,
    control_label: str | None = None,
    other_label: str | None = None,
    rise_rank: int = 25,
    cp_rank: int = 10,
    lr_pairs: pd.DataFrame | None = None,
    rise_n_iter_max: int = 100,
    rise_tol: float = 1e-6,
    cp_n_iter_max: int = 100,
    cp_tol: float = 1e-3,
    init: str = "svd",
    svd: str = "randomized_svd",
    complex_sep: str | None = "&",
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
    random_state: int | None = None,
    backend: str | None = None,
    compress: bool = True,
    oversample: int = 5,
) -> dict[str, Any]:
    """Run RISE, the interaction tensor and CP over `adata`, then test the
    resulting condition factors against the phenotype by one-way ANOVA.

    Parameters
    ----------
    adata : anndata.AnnData
        Data to fit -- the full cohort for the reference fit, a resampled
        roster for a replicate.
    condition_column, group_column : str
        Columns in `obs` holding the patient identity and its group label.
    control_label, other_label : str | None, optional
        Values of `group_column` collapsed to the reference and other level.
        Both None -- the default -- tests every level of `group_column`; see
        `anova.prepare_anova_inputs`.
    rise_rank, cp_rank : int
        The already-selected ranks the bootstrap is conditional on.
    lr_pairs : pd.DataFrame | None, optional
        Ligand-receptor pairs; loaded from the packaged database if None.
    rise_n_iter_max, rise_tol : int, float
        Fitting controls for the RISE fit.
    cp_n_iter_max, cp_tol, init, svd : int, float, str, str
        Fitting controls for the CP fit.
    complex_sep, communication_score, hill_k : str | None, str, float
        Communication-tensor options; see `ccc_rise.calc_communication_score`.
    random_state : int | None, optional
        Seed for the RISE fit and the CP fit. The ANOVA is deterministic.
    backend : str | None, default=None
        Backend the RISE fit runs on ('cpu', 'cupy', 'mlx', or None to
        auto-detect); passed straight through to `tensor.rise_store_r2x`.
    compress : bool, default=True
        Fit CP inside a CANDELINC core sized for `cp_rank` instead of the
        full tensor (see `stability.component_stability._fit_cp`).
    oversample : int, default=5
        Extra basis columns per mode beyond `cp_rank`, when `compress`.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys 'factors', 'cp_weights', 'lr_symbols', 'anova'
        (the per-component table from `anova.anova_component_associations`),
        'r2x_cp', 'sample_to_group'.
    """
    if lr_pairs is None:
        from ..import_data import import_ligand_receptor_pairs

        lr_pairs = import_ligand_receptor_pairs()

    adata = add_cond_idxs(adata, condition_column)
    adata, _r2x_rise = rise_store_r2x(
        adata,
        rank=rise_rank,
        n_iter_max=rise_n_iter_max,
        tolerance=rise_tol,
        random_state=random_state,
        backend=backend,
    )
    tensor, filtered_lr_pairs = calculate_interaction_tensor(
        adata,
        lr_pairs,
        complex_sep=complex_sep,
        communication_score=communication_score,
        hill_k=hill_k,
    )
    validate_cp_rank(cp_rank, tensor.shape)

    cp_weights, cp_factors, r2x_cp, _diagnostics = _fit_cp(
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
    cp_weights, factors = standardize_cc_pf2(cp_weights, cp_factors)

    X, sample_to_group = prepare_anova_inputs(
        adata,
        factors,
        condition_column,
        group_column,
        control_label,
        other_label,
    )
    anova = anova_component_associations(
        X, sample_to_group, control_label=control_label
    )
    return {
        "factors": factors,
        "cp_weights": cp_weights,
        "lr_symbols": filtered_lr_pairs["interaction_symbol"].to_numpy(),
        "anova": anova,
        "r2x_cp": r2x_cp,
        "sample_to_group": sample_to_group,
    }


#: Column order of one replicate's table, and so of the replicate-level CSV
#: once `bootstrap_anova` prepends 'Replicate'.
REPLICATE_COLUMNS = [
    "Component",
    "MatchedComponent",
    "EtaSquared",
    "F",
    "P",
    "GroupMeanDiff",
    "MatchScore",
    "SignedCosine",
    "RunnerUpScore",
    "R2XCP",
]


def bootstrap_replicate_anova(
    adata: anndata.AnnData,
    reference: dict[str, Any],
    condition_column: str,
    group_column: str,
    control_label: str | None = None,
    other_label: str | None = None,
    rise_rank: int = 25,
    cp_rank: int = 10,
    lr_pairs: pd.DataFrame | None = None,
    rise_n_iter_max: int = 100,
    rise_tol: float = 1e-6,
    cp_n_iter_max: int = 100,
    cp_tol: float = 1e-3,
    init: str = "svd",
    svd: str = "randomized_svd",
    complex_sep: str | None = "&",
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
    stratify: bool = True,
    patient_column: str | None = None,
    random_state: int | None = None,
    backend: str | None = None,
    compress: bool = True,
    oversample: int = 5,
) -> pd.DataFrame:
    """One bootstrap replicate: resample patients, re-run the pipeline, and
    report each reference component's ANOVA against the phenotype from the
    resampled fit.

    Parameters
    ----------
    adata : anndata.AnnData
        Full-cohort data to resample patients from.
    reference : dict[str, Any]
        The full-cohort fit whose component numbering and orientation the
        replicate's statistics are expressed in.
    condition_column, group_column : str
        Columns in `obs` holding the patient identity and its group label.
    control_label, other_label : str | None, optional
        Values of `group_column` collapsed to the reference and other level;
        both None tests every level. See `anova.prepare_anova_inputs`.
    rise_rank, cp_rank : int
        The already-selected ranks the bootstrap is conditional on.
    lr_pairs : pd.DataFrame | None, optional
        Ligand-receptor pairs; loaded from the packaged database if None.
    rise_n_iter_max, rise_tol : int, float
        Fitting controls for the RISE fit.
    cp_n_iter_max, cp_tol, init, svd : int, float, str, str
        Fitting controls for the CP fit.
    complex_sep, communication_score, hill_k : str | None, str, float
        Communication-tensor options; see `ccc_rise.calc_communication_score`.
    stratify : bool, default=True
        Draw patients within each `group_column` stratum, keeping the class
        balance fixed; see `utils.resample_conditions`.
    patient_column : str | None, optional
        Condition-level patient identity column. Drawing one of a
        multi-sample patient's conditions carries its sibling in too; see
        `utils.resample_conditions`'s `patient_by`.
    random_state : int | None, optional
        Seed for the draw and for the replicate's fits.
    backend : str | None, default=None
        Backend the replicate's RISE refit runs on; see `fit_anova_pipeline`.
    compress : bool, default=True
        Fit the replicate's CP stage inside a CANDELINC core; see
        `stability.component_stability._fit_cp`.
    oversample : int, default=5
        Extra basis columns per mode beyond `cp_rank`, when `compress`.

    Returns
    -------
    pd.DataFrame
        One row per reference component, columns `REPLICATE_COLUMNS`.
        'GroupMeanDiff' is sign-corrected onto the reference component's
        orientation; 'EtaSquared', 'F' and 'P' are sign-invariant and are
        carried across on the matched component alone.
    """
    resampled = resample_conditions(
        adata,
        condition_column,
        stratify_by=group_column if stratify else None,
        patient_by=patient_column,
        random_seed=random_state,
    )
    fit = fit_anova_pipeline(
        resampled,
        lr_pairs=lr_pairs,
        condition_column=condition_column,
        group_column=group_column,
        control_label=control_label,
        other_label=other_label,
        rise_rank=rise_rank,
        cp_rank=cp_rank,
        rise_n_iter_max=rise_n_iter_max,
        rise_tol=rise_tol,
        cp_n_iter_max=cp_n_iter_max,
        cp_tol=cp_tol,
        init=init,
        svd=svd,
        complex_sep=complex_sep,
        communication_score=communication_score,
        hill_k=hill_k,
        random_state=random_state,
        backend=backend,
        compress=compress,
        oversample=oversample,
    )

    if not np.array_equal(fit["lr_symbols"], reference["lr_symbols"]):
        raise ValueError(
            "Replicate and reference fits retained different ligand-receptor "
            "pairs, so their LR-pair factor rows are not comparable and "
            "component matching would be meaningless."
        )

    anova = fit["anova"]
    aligned = align_to_reference_components(
        reference["factors"],
        fit["factors"],
        anova["GroupMeanDiff"].to_numpy(dtype=float),
        mode_weights=BOOTSTRAP_MATCH_MODE_WEIGHTS,
    ).rename(columns={"Value": "GroupMeanDiff"})

    # F, P and eta-squared do not change under a gauge flip, so they are read
    # straight off the replicate component the alignment picked out.
    matched = aligned["MatchedComponent"].to_numpy(dtype=int) - 1
    for column in ("EtaSquared", "F", "P"):
        aligned[column] = anova[column].to_numpy(dtype=float)[matched]

    aligned["R2XCP"] = fit["r2x_cp"]
    return aligned[REPLICATE_COLUMNS]


def bootstrap_anova(
    adata: anndata.AnnData,
    reference: dict[str, Any],
    condition_column: str,
    group_column: str,
    control_label: str | None = None,
    other_label: str | None = None,
    rise_rank: int = 25,
    cp_rank: int = 10,
    n_boot: int = 100,
    lr_pairs: pd.DataFrame | None = None,
    rise_n_iter_max: int = 100,
    rise_tol: float = 1e-6,
    cp_n_iter_max: int = 100,
    cp_tol: float = 1e-3,
    init: str = "svd",
    svd: str = "randomized_svd",
    complex_sep: str | None = "&",
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
    stratify: bool = True,
    patient_column: str | None = None,
    random_state: int = 0,
    backend: str | None = None,
    compress: bool = True,
    oversample: int = 5,
) -> pd.DataFrame:
    """Run `n_boot` patient-resampled replicates of the pipeline.

    Parameters
    ----------
    adata : anndata.AnnData
        Full-cohort data to resample patients from.
    reference : dict[str, Any]
        The full-cohort fit dictionary returned by `fit_anova_pipeline`, whose
        component numbering and orientation the replicates are expressed in.
    condition_column, group_column : str
        Columns in `obs` holding the patient identity and its group label.
    control_label, other_label : str | None, optional
        Values of `group_column` collapsed to the reference and other level;
        both None tests every level. See `anova.prepare_anova_inputs`.
    rise_rank, cp_rank : int
        The already-selected ranks the bootstrap is conditional on.
    n_boot : int, default=100
        Number of replicates.
    lr_pairs : pd.DataFrame | None, optional
        Ligand-receptor pairs; loaded from the packaged database if None.
    rise_n_iter_max, rise_tol : int, float
        Fitting controls for RISE.
    cp_n_iter_max, cp_tol, init, svd : int, float, str, str
        Fitting controls for CP.
    complex_sep, communication_score, hill_k : str | None, str, float
        Communication-tensor options; see `ccc_rise.calc_communication_score`.
    stratify : bool, default=True
        Draw patients within each `group_column` stratum, keeping the class
        balance fixed; see `utils.resample_conditions`.
    patient_column : str | None, optional
        Condition-level patient identity column. Drawing one of a
        multi-sample patient's conditions carries its sibling in too; see
        `utils.resample_conditions`'s `patient_by`.
    random_state : int, default=0
        Base seed; replicate ``b`` uses `replicate_seed(random_state, b)`.
    backend : str | None, default=None
        Backend every replicate's RISE refit runs on; see `fit_anova_pipeline`.
    compress : bool, default=True
        Fit every replicate's CP stage inside a CANDELINC core; see
        `stability.component_stability._fit_cp`.
    oversample : int, default=5
        Extra basis columns per mode beyond `cp_rank`, when `compress`.

    Returns
    -------
    pd.DataFrame
        `bootstrap_replicate_anova` stacked over replicates, with a leading
        'Replicate' column.
    """
    frames = []
    for b in range(n_boot):
        print(f"   replicate {b + 1}/{n_boot}")
        seed = replicate_seed(random_state, b)
        replicate = bootstrap_replicate_anova(
            adata,
            reference=reference,
            condition_column=condition_column,
            group_column=group_column,
            control_label=control_label,
            other_label=other_label,
            rise_rank=rise_rank,
            cp_rank=cp_rank,
            lr_pairs=lr_pairs,
            rise_n_iter_max=rise_n_iter_max,
            rise_tol=rise_tol,
            cp_n_iter_max=cp_n_iter_max,
            cp_tol=cp_tol,
            init=init,
            svd=svd,
            complex_sep=complex_sep,
            communication_score=communication_score,
            hill_k=hill_k,
            stratify=stratify,
            patient_column=patient_column,
            random_state=seed,
            backend=backend,
            compress=compress,
            oversample=oversample,
        )
        replicate.insert(0, "Replicate", b)
        frames.append(replicate)
    return pd.concat(frames, ignore_index=True)


def summarize_bootstrap_anova(
    boot_anova: pd.DataFrame,
    reference_anova: pd.DataFrame,
    ci_level: float = 95.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One row per component: its bootstrap effect-size distribution, and how
    reliably the association and its direction survive resampling.

    Three things are reported side by side rather than collapsed into one
    number, because they call for different conclusions: how large the effect
    is and how tightly the rosters pin it down (the eta-squared interval); how
    often the association reaches `alpha` at all (`FracSignificant`); and
    whether the component was even re-identifiable across replicates
    (`MedianMatchScore`/`MinMatchScore`), which is a CP identifiability
    question rather than a question about the phenotype.

    There is deliberately no "interval excludes zero" column: eta-squared is a
    variance ratio and cannot be negative, so such a test would be vacuous.
    `FracSignificant` takes its place -- read as a descriptive tally, per the
    degrees-of-freedom caveat in this module's docstring.

    Parameters
    ----------
    boot_anova : pd.DataFrame
        Output of `bootstrap_anova`.
    reference_anova : pd.DataFrame
        The full-cohort fit's per-component ANOVA table -- `fit_anova_pipeline`'s
        ``result["anova"]`` -- carrying 'Component', 'EtaSquared', 'F', 'P' and
        'GroupMeanDiff'.
    ci_level : float, default=95.0
        Central percentile interval width, in percent, taken on eta-squared.
    alpha : float, default=0.05
        Threshold `FracSignificant` counts replicates against.

    Returns
    -------
    pd.DataFrame
        One row per component, sorted by component, with columns 'Component',
        'ReferenceEtaSquared', 'ReferenceF', 'ReferenceP', 'MeanEtaSquared',
        'MedianEtaSquared', 'SDEtaSquared', 'CILow', 'CIHigh', 'MedianP',
        'FracSignificant', 'ReferenceGroupMeanDiff', 'MedianGroupMeanDiff',
        'FracSameSignAsReference' (NaN when the reference group-mean
        difference is exactly zero, i.e. has no direction to agree with),
        'MedianMatchScore', 'MinMatchScore', 'NReplicates'.
    """
    tail = (100.0 - ci_level) / 2.0
    reference_by_component = reference_anova.set_index("Component")
    rows = []
    for component_key, group in boot_anova.groupby("Component"):
        component = cast(int, component_key)
        eta = group["EtaSquared"].to_numpy(dtype=float)
        p_values = group["P"].to_numpy(dtype=float)
        diffs = group["GroupMeanDiff"].to_numpy(dtype=float)
        reference = reference_by_component.loc[component]
        reference_diff = float(reference["GroupMeanDiff"])
        ci_low, ci_high = np.nanpercentile(eta, [tail, 100.0 - tail])
        rows.append(
            {
                "Component": component,
                "ReferenceEtaSquared": float(reference["EtaSquared"]),
                "ReferenceF": float(reference["F"]),
                "ReferenceP": float(reference["P"]),
                "MeanEtaSquared": float(np.nanmean(eta)),
                "MedianEtaSquared": float(np.nanmedian(eta)),
                "SDEtaSquared": float(np.nanstd(eta, ddof=1)) if len(eta) > 1 else 0.0,
                "CILow": float(ci_low),
                "CIHigh": float(ci_high),
                "MedianP": float(np.nanmedian(p_values)),
                # A replicate whose ANOVA came back degenerate (NaN p) counts
                # as not significant rather than dropping out of the tally.
                "FracSignificant": float(np.mean(p_values < alpha)),
                "ReferenceGroupMeanDiff": reference_diff,
                "MedianGroupMeanDiff": float(np.nanmedian(diffs)),
                "FracSameSignAsReference": (
                    float(np.mean(np.sign(diffs) == np.sign(reference_diff)))
                    if np.isfinite(reference_diff) and reference_diff != 0
                    else np.nan
                ),
                "MedianMatchScore": float(np.median(group["MatchScore"])),
                "MinMatchScore": float(np.min(group["MatchScore"])),
                "NReplicates": len(eta),
            }
        )
    return pd.DataFrame(rows).sort_values("Component").reset_index(drop=True)
