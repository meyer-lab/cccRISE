"""
Patient-level bootstrap of the CCC-RISE logistic regression weights.

A single logistic regression fit on one CCC-RISE factorization reports a
weight per CP component, but says nothing about how much of that weight is
attributable to *which patients happened to be enrolled*. Resampling the
patient roster with replacement and re-running the whole pipeline -- RISE, the
interaction tensor, CP at the same fixed ranks, then the logistic regression
-- turns each component's weight into a distribution whose spread is exactly
that sampling variability.

Every replicate refits RISE and CP from scratch, so its components come out in
an arbitrary order and sign; each replicate is therefore matched back onto the
full-cohort ("reference") fit's components before its weights are recorded
(`align_to_reference_components`). The match is computed on the
ligand-receptor mode alone, the one mode whose coordinates mean the same thing
in two fits with separately-fit RISE factors, and its score is kept alongside
every weight: a component that could not be re-identified in a replicate
produces a wide weight distribution for a reason that has nothing to do with
the logistic regression, and the two explanations have to stay separable.
"""

from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..logreg import fit_logreg_weights, prepare_logreg_inputs
from ..tensor import fit_rise_cp_pipeline
from ..utils import replicate_seed, resample_conditions

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


def fit_logreg_pipeline(
    adata: anndata.AnnData,
    condition_column: str,
    group_column: str,
    control_label: str,
    other_label: str,
    rise_rank: int,
    cp_rank: int,
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
    scoring: str = "roc_auc",
    random_state: int | None = None,
) -> dict[str, Any]:
    """Run RISE, the interaction tensor and CP over `adata`, then fit the
    logistic regression on the resulting condition factors.

    Parameters
    ----------
    adata : anndata.AnnData
        Data to fit -- the full cohort for the reference fit, a resampled
        roster for a replicate.
    condition_column, group_column : str
        Columns in `obs` holding the patient identity and its group label.
    control_label : str
        Value of `group_column` mapped to the reference class.
    other_label : str
        Label applied to every other value of `group_column`.
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
    scoring : str, default="roc_auc"
        Scikit-learn scoring the logistic regression selects its
        regularization strength by; see `logreg.fit_logreg_weights`.
    random_state : int | None, optional
        Seed for the RISE fit, the CP fit and the logistic regression's
        cross-validation.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys 'factors', 'cp_weights', 'lr_symbols',
        'logreg_weights', 'r2x_cp', 'sample_to_group'.
    """
    if lr_pairs is None:
        from ..import_data import import_ligand_receptor_pairs

        lr_pairs = import_ligand_receptor_pairs()

    fit = fit_rise_cp_pipeline(
        adata,
        lr_pairs,
        condition_column,
        rise_rank,
        cp_rank,
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
    )
    adata = fit["adata"]
    filtered_lr_pairs = fit["filtered_lr_pairs"]
    cp_weights, factors, r2x_cp = fit["cp_weights"], fit["cp_factors"], fit["r2x_cp"]

    X, sample_to_group = prepare_logreg_inputs(
        adata,
        factors,
        condition_column,
        group_column,
        control_label,
        other_label,
    )
    logreg_weights = fit_logreg_weights(
        X,
        sample_to_group,
        scoring=scoring,
        random_state=random_state,
    )
    return {
        "factors": factors,
        "cp_weights": cp_weights,
        "lr_symbols": filtered_lr_pairs["interaction_symbol"].to_numpy(),
        "logreg_weights": logreg_weights,
        "r2x_cp": r2x_cp,
        "sample_to_group": sample_to_group,
    }


def bootstrap_replicate_weights(
    adata: anndata.AnnData,
    reference: dict[str, Any],
    condition_column: str,
    group_column: str,
    control_label: str,
    other_label: str,
    rise_rank: int,
    cp_rank: int,
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
    scoring: str = "roc_auc",
    stratify: bool = True,
    random_state: int | None = None,
) -> pd.DataFrame:
    """One bootstrap replicate: resample patients, re-run the pipeline, and
    report each reference component's logistic regression weight from the
    resampled fit.

    Parameters
    ----------
    adata : anndata.AnnData
        Full-cohort data to resample patients from.
    reference : dict[str, Any]
        The full-cohort fit whose component numbering and orientation the
        replicate's weights are expressed in.
    condition_column, group_column : str
        Columns in `obs` holding the patient identity and its group label.
    control_label : str
        Value of `group_column` mapped to the reference class.
    other_label : str
        Label applied to every other value of `group_column`.
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
    scoring : str, default="roc_auc"
        Scikit-learn scoring the logistic regression selects its
        regularization strength by; see `logreg.fit_logreg_weights`.
    stratify : bool, default=True
        Draw patients within each `group_column` stratum, keeping the class
        balance fixed; see `utils.resample_conditions`.
    random_state : int | None, optional
        Seed for the draw and for the replicate's fits.

    Returns
    -------
    pd.DataFrame
        One row per reference component, columns 'Component',
        'MatchedComponent', 'Weight' (sign-corrected onto the reference
        component's orientation), 'MatchScore', 'SignedCosine',
        'RunnerUpScore', 'R2XCP'.
    """
    resampled = resample_conditions(
        adata,
        condition_column,
        stratify_by=group_column if stratify else None,
        random_seed=random_state,
    )
    fit = fit_logreg_pipeline(
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
        scoring=scoring,
        random_state=random_state,
    )

    if not np.array_equal(fit["lr_symbols"], reference["lr_symbols"]):
        raise ValueError(
            "Replicate and reference fits retained different ligand-receptor "
            "pairs, so their LR-pair factor rows are not comparable and "
            "component matching would be meaningless."
        )

    aligned = align_to_reference_components(
        reference["factors"],
        fit["factors"],
        fit["logreg_weights"],
        mode_weights=BOOTSTRAP_MATCH_MODE_WEIGHTS,
    ).rename(columns={"Value": "Weight"})
    aligned["R2XCP"] = fit["r2x_cp"]
    return aligned


def bootstrap_logreg_weights(
    adata: anndata.AnnData,
    reference: dict[str, Any],
    condition_column: str,
    group_column: str,
    control_label: str,
    other_label: str,
    rise_rank: int,
    cp_rank: int,
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
    scoring: str = "roc_auc",
    stratify: bool = True,
    random_state: int = 0,
) -> pd.DataFrame:
    """Run `n_boot` patient-resampled replicates of the pipeline.

    Parameters
    ----------
    adata : anndata.AnnData
        Full-cohort data to resample patients from.
    reference : dict[str, Any]
        The full-cohort fit dictionary returned by `fit_logreg_pipeline`, whose
        component numbering and orientation the replicate's weights are
        expressed in.
    condition_column, group_column : str
        Columns in `obs` holding the patient identity and its group label.
    control_label : str
        Value of `group_column` mapped to the reference class.
    other_label : str
        Label applied to every other value of `group_column`.
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
    scoring : str, default="roc_auc"
        Scikit-learn scoring the logistic regression selects its
        regularization strength by; see `logreg.fit_logreg_weights`.
    stratify : bool, default=True
        Draw patients within each `group_column` stratum, keeping the class
        balance fixed; see `utils.resample_conditions`.
    random_state : int, default=0
        Base seed; replicate ``b`` uses `replicate_seed(random_state, b)`.

    Returns
    -------
    pd.DataFrame
        `bootstrap_replicate_weights` stacked over replicates, with a
        leading 'Replicate' column.
    """
    frames = []
    for b in range(n_boot):
        print(f"   replicate {b + 1}/{n_boot}")
        seed = replicate_seed(random_state, b)
        replicate = bootstrap_replicate_weights(
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
            scoring=scoring,
            stratify=stratify,
            random_state=seed,
        )
        replicate.insert(0, "Replicate", b)
        frames.append(replicate)
    return pd.concat(frames, ignore_index=True)


def summarize_bootstrap_weights(
    boot_weights: pd.DataFrame,
    reference_weights: np.ndarray,
    ci_level: float = 95.0,
) -> pd.DataFrame:
    """One row per component: its bootstrap weight distribution, and how
    reliably the sign and the selection survive resampling.

    Three failure modes are reported side by side rather than collapsed into
    one number, because they call for different conclusions: a weight whose
    interval spans zero is not distinguishable from no association at all; a
    weight that is frequently regularized to exactly zero was selected by the
    L1 penalty only for some rosters; and a component with a low match score
    was not reliably re-identified across replicates, so its spread says more
    about CP identifiability than about the regression.

    Parameters
    ----------
    boot_weights : pd.DataFrame
        Output of `bootstrap_logreg_weights`.
    reference_weights : np.ndarray
        The full-cohort fit's per-component weights, indexed from component 1
        in order.
    ci_level : float, default=95.0
        Central percentile interval width, in percent.

    Returns
    -------
    pd.DataFrame
        One row per component, sorted by component, with columns 'Component',
        'ReferenceWeight', 'MeanWeight', 'MedianWeight', 'SDWeight', 'CILow',
        'CIHigh', 'CIExcludesZero', 'FracNonzero',
        'FracSameSignAsReference' (NaN when the reference weight is exactly
        zero, i.e. has no sign to agree with), 'MedianMatchScore',
        'MinMatchScore', 'NReplicates'.
    """
    tail = (100.0 - ci_level) / 2.0
    rows = []
    for component_key, group in boot_weights.groupby("Component"):
        component = cast(int, component_key)
        weights = group["Weight"].to_numpy()
        reference = float(reference_weights[component - 1])
        ci_low, ci_high = np.percentile(weights, [tail, 100.0 - tail])
        rows.append(
            {
                "Component": component,
                "ReferenceWeight": reference,
                "MeanWeight": float(np.mean(weights)),
                "MedianWeight": float(np.median(weights)),
                "SDWeight": float(np.std(weights, ddof=1)) if len(weights) > 1 else 0.0,
                "CILow": float(ci_low),
                "CIHigh": float(ci_high),
                "CIExcludesZero": bool(ci_low > 0 or ci_high < 0),
                "FracNonzero": float(np.mean(weights != 0)),
                "FracSameSignAsReference": (
                    float(np.mean(np.sign(weights) == np.sign(reference)))
                    if reference != 0
                    else np.nan
                ),
                "MedianMatchScore": float(np.median(group["MatchScore"])),
                "MinMatchScore": float(np.min(group["MatchScore"])),
                "NReplicates": len(weights),
            }
        )
    return pd.DataFrame(rows).sort_values("Component").reset_index(drop=True)
