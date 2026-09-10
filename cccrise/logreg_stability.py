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

from dataclasses import dataclass
from typing import NamedTuple, cast

import anndata
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .ccc_rise import cp_decomposition, standardize_cc_pf2, validate_cp_rank
from .import_data import add_cond_idxs
from .logreg import fit_logreg_weights, prepare_logreg_inputs
from .tensor import calculate_interaction_tensor, rise_store_r2x
from .utils import resample_conditions

Factors = list[np.ndarray]

#: Modes `align_to_reference_components` matches a replicate to the reference
#: over. Only the ligand-receptor mode (3) survives a separate RISE fit with
#: its coordinates intact: the condition mode indexes a different patient
#: roster, and the sender/receiver modes are expressed in that replicate's own
#: RISE eigen-states.
BOOTSTRAP_MATCH_MODE_WEIGHTS: dict[int, float] = {3: 1.0}


def replicate_seed(random_state: int, replicate: int) -> int:
    """Seed for one replicate: ``random_state * 1000 + replicate + 1``."""
    return random_state * 1000 + replicate + 1


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


class LogRegFit(NamedTuple):
    """One pass of the pipeline: the CP fit and the logistic regression on it."""

    factors: Factors
    cp_weights: np.ndarray
    lr_symbols: np.ndarray
    logreg_weights: np.ndarray
    r2x_cp: float


@dataclass(frozen=True)
class LogRegStabilitySpec:
    """The pipeline configuration a bootstrap holds fixed across replicates.

    Bundled rather than passed replicate by replicate because the reference
    fit and every resampled fit have to agree on all of it -- differing ranks
    or a differing group encoding would make their weights incomparable, which
    is the one thing a bootstrap distribution cannot tolerate.

    Attributes
    ----------
    condition_column, group_column : str
        Columns in `obs` holding the patient identity and its group label.
    control_label : str
        Value of `group_column` mapped to the reference class.
    other_label : str
        Label applied to every other value of `group_column`.
    rise_rank, cp_rank : int
        The already-selected ranks the bootstrap is conditional on.
    rise_n_iter_max, rise_tol : int, float
        Fitting controls for the RISE fit.
    cp_n_iter_max, cp_tol, init, svd : int, float, str, str
        Fitting controls for the CP fit.
    complex_sep, communication_score, hill_k : str | None, str, float
        Communication-tensor options; see `ccc_rise.calc_communication_score`.
    scoring : str
        Scikit-learn scoring the logistic regression selects its
        regularization strength by; see `logreg.fit_logreg_weights`.
    """

    condition_column: str
    group_column: str
    control_label: str
    other_label: str
    rise_rank: int
    cp_rank: int
    rise_n_iter_max: int = 100
    rise_tol: float = 1e-6
    cp_n_iter_max: int = 100
    cp_tol: float = 1e-3
    init: str = "svd"
    svd: str = "randomized_svd"
    complex_sep: str | None = "&"
    communication_score: str = "expression_product"
    hill_k: float = 0.5
    scoring: str = "roc_auc"

    def fit(
        self,
        adata: anndata.AnnData,
        lr_pairs: pd.DataFrame | None = None,
        random_state: int | None = None,
    ) -> LogRegFit:
        """Run RISE, the interaction tensor and CP over `adata`, then fit the
        logistic regression on the resulting condition factors.

        Parameters
        ----------
        adata : anndata.AnnData
            Data to fit -- the full cohort for the reference fit, a resampled
            roster for a replicate.
        lr_pairs : pd.DataFrame | None, optional
            Ligand-receptor pairs; loaded from the packaged database if None.
        random_state : int | None, optional
            Seed for the RISE fit, the CP fit and the logistic regression's
            cross-validation.

        Returns
        -------
        LogRegFit
        """
        if lr_pairs is None:
            from .import_data import import_ligand_receptor_pairs

            lr_pairs = import_ligand_receptor_pairs()

        adata = add_cond_idxs(adata, self.condition_column)
        adata, _r2x_rise = rise_store_r2x(
            adata,
            rank=self.rise_rank,
            n_iter_max=self.rise_n_iter_max,
            tolerance=self.rise_tol,
            random_state=random_state,
        )
        tensor, filtered_lr_pairs = calculate_interaction_tensor(
            adata,
            lr_pairs,
            complex_sep=self.complex_sep,
            communication_score=self.communication_score,
            hill_k=self.hill_k,
        )
        validate_cp_rank(self.cp_rank, tensor.shape)

        cp_weights, cp_factors, r2x_cp = cp_decomposition(
            tensor,
            self.cp_rank,
            n_iter_max=self.cp_n_iter_max,
            tol=self.cp_tol,
            init=self.init,
            svd=self.svd,
            random_state=random_state,
        )
        cp_weights, factors = standardize_cc_pf2(cp_weights, cp_factors)

        X, sample_to_group = prepare_logreg_inputs(
            adata,
            factors,
            self.condition_column,
            self.group_column,
            self.control_label,
            self.other_label,
        )
        logreg_weights = fit_logreg_weights(
            X,
            sample_to_group,
            scoring=self.scoring,
            random_state=random_state,
        )
        return LogRegFit(
            factors=factors,
            cp_weights=cp_weights,
            lr_symbols=filtered_lr_pairs["interaction_symbol"].to_numpy(),
            logreg_weights=logreg_weights,
            r2x_cp=r2x_cp,
        )


def bootstrap_replicate_weights(
    spec: LogRegStabilitySpec,
    adata: anndata.AnnData,
    reference: LogRegFit,
    lr_pairs: pd.DataFrame | None = None,
    stratify: bool = True,
    random_state: int | None = None,
) -> pd.DataFrame:
    """One bootstrap replicate: resample patients, re-run the pipeline, and
    report each reference component's logistic regression weight from the
    resampled fit.

    Parameters
    ----------
    spec : LogRegStabilitySpec
        Configuration shared with the reference fit.
    adata : anndata.AnnData
        Full-cohort data to resample patients from.
    reference : LogRegFit
        The full-cohort fit whose component numbering and orientation the
        replicate's weights are expressed in.
    lr_pairs : pd.DataFrame | None, optional
        Ligand-receptor pairs; loaded from the packaged database if None.
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
        spec.condition_column,
        stratify_by=spec.group_column if stratify else None,
        random_seed=random_state,
    )
    fit = spec.fit(resampled, lr_pairs=lr_pairs, random_state=random_state)

    if not np.array_equal(fit.lr_symbols, reference.lr_symbols):
        raise ValueError(
            "Replicate and reference fits retained different ligand-receptor "
            "pairs, so their LR-pair factor rows are not comparable and "
            "component matching would be meaningless."
        )

    aligned = align_to_reference_components(
        reference.factors,
        fit.factors,
        fit.logreg_weights,
        mode_weights=BOOTSTRAP_MATCH_MODE_WEIGHTS,
    ).rename(columns={"Value": "Weight"})
    aligned["R2XCP"] = fit.r2x_cp
    return aligned


def bootstrap_logreg_weights(
    spec: LogRegStabilitySpec,
    adata: anndata.AnnData,
    reference: LogRegFit,
    n_boot: int = 100,
    lr_pairs: pd.DataFrame | None = None,
    stratify: bool = True,
    random_state: int = 0,
) -> pd.DataFrame:
    """Run `n_boot` patient-resampled replicates of the pipeline.

    Parameters
    ----------
    spec, adata, reference, lr_pairs, stratify
        See `bootstrap_replicate_weights`.
    n_boot : int, default=100
        Number of replicates.
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
        replicate = bootstrap_replicate_weights(
            spec,
            adata,
            reference,
            lr_pairs=lr_pairs,
            stratify=stratify,
            random_state=replicate_seed(random_state, b),
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
