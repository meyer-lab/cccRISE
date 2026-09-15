"""One-way ANOVA of CP condition-factor loadings against a phenotype.

Replaces the L1 logistic regression that used to score `X.uns["A"]`.
The grouping is carried as a named label per condition rather than as an
integer code: `prepare_anova_inputs` collapses to two named levels when told
to, and otherwise passes every level of the group column through. Moving from
control-vs-ALAD to three-level `ALADstatus`, or to Died/CLAD/No CLAD, is then
an argument change rather than a code change.

p-values are reported raw, with no multiple-testing correction across
components.
"""

import anndata
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from .figures.commonFuncs.plotGeneral import collapse_group, get_sample_to_group
from .utils import correct_conditions


def prepare_anova_inputs(
    adata: anndata.AnnData,
    factors: list[np.ndarray],
    condition_column: str,
    group_column: str,
    control_label: str | None = None,
    other_label: str | None = None,
) -> tuple[anndata.AnnData, pd.Series]:
    """Attach one CP fit's factors to an AnnData object and derive the group
    label the ANOVA is run against.

    Mirrors the condition-factor read-depth correction and group labeling
    `figures/figure4.py` applies to the primary post hoc factorization, so
    results computed from any other fit -- a different rank, a bootstrap
    replicate -- are directly comparable to the ones reported there.

    Parameters
    ----------
    adata : anndata.AnnData
        Data the fit came from, carrying `condition_unique_idxs` in `obs`.
    factors : list[np.ndarray]
        Standardized CP factors [A, B, C, D].
    condition_column, group_column : str
        Columns in `obs` holding the condition identity and its group label.
    control_label, other_label : str | None, optional
        When both are given, `group_column` is collapsed to those two named
        levels. When both are None -- the default -- every level of
        `group_column` is passed through untouched, which is the k-level
        ANOVA path.

    Returns
    -------
    tuple[anndata.AnnData, pd.Series]
        A copy carrying the factors in `uns` with `uns["A"]` read-depth
        corrected, and the group label *name* per condition, aligned to
        `uns["A"]`'s rows.
    """
    X = adata.copy()
    X.uns["A"], X.uns["B"], X.uns["C"], X.uns["D"] = factors
    X.uns["A"] = correct_conditions(X)

    sample_to_group = get_sample_to_group(X, condition_column, group_column)
    if control_label is None and other_label is None:
        return X, sample_to_group.astype(str)
    if control_label is None or other_label is None:
        raise ValueError(
            "Pass both control_label and other_label to collapse to two named "
            f"levels, or neither to test every level of {group_column!r}; got "
            f"control_label={control_label!r}, other_label={other_label!r}."
        )
    return X, collapse_group(sample_to_group, control_label, control_label, other_label)


def anova_component_associations(
    X: anndata.AnnData,
    sample_to_group: pd.Series,
    alpha: float = 0.05,
    control_label: str | None = None,
) -> pd.DataFrame:
    """One-way ANOVA of each CP component's condition loading against the
    group label.

    Parameters
    ----------
    X : anndata.AnnData
        Carries the condition factor matrix in `uns["A"]`, one row per
        condition in `condition_unique_idxs` order.
    sample_to_group : pd.Series
        Group label name per condition, aligned to `uns["A"]`'s rows.
    alpha : float, default=0.05
        Threshold `Significant` is reported against. The test itself is
        unaffected, and the p-values stay uncorrected.
    control_label : str | None, optional
        Level `GroupMeanDiff` is measured *from*. Defaults to the
        alphabetically first level, which is the level `binarize_group`'s
        codes made class 0, so the sign keeps the meaning it had under the
        logistic regression.

    Returns
    -------
    pd.DataFrame
        One row per component (1-indexed), columns 'Component', 'F', 'P',
        'EtaSquared', 'GroupMeanDiff', 'Significant', 'NGroups', 'GroupSizes'.
        A component whose grouping is degenerate -- fewer than two levels, or
        any level with fewer than two samples -- reports NaN F/P/eta-squared
        rather than raising, so one bad component cannot abort a whole
        bootstrap replicate.
    """
    A = np.asarray(X.uns["A"], dtype=float)
    labels = pd.Series(np.asarray(sample_to_group), dtype=object).reset_index(drop=True)

    rows = []
    for component in range(1, A.shape[1] + 1):
        values = A[:, component - 1]
        # A missing label or a non-finite loading has no group to sit in; drop
        # that condition rather than let it poison the component's whole test.
        keep = labels.notna().to_numpy() & np.isfinite(values)
        kept_labels = labels[keep].to_numpy()
        kept_values = values[keep]

        level_names = sorted({str(name) for name in kept_labels})
        groups = [kept_values[kept_labels == name] for name in level_names]
        group_sizes = [int(group.size) for group in groups]

        if len(groups) < 2 or any(size < 2 for size in group_sizes):
            f_stat = p_value = eta_squared = group_mean_diff = np.nan
        else:
            f_stat, p_value = (float(stat) for stat in f_oneway(*groups))
            eta_squared = _eta_squared(groups)
            group_mean_diff = _group_mean_diff(level_names, groups, control_label)

        rows.append(
            {
                "Component": component,
                "F": f_stat,
                "P": p_value,
                "EtaSquared": eta_squared,
                "GroupMeanDiff": group_mean_diff,
                "Significant": bool(p_value < alpha) if np.isfinite(p_value) else False,
                "NGroups": len(groups),
                "GroupSizes": group_sizes,
            }
        )
    return pd.DataFrame(rows)


def _eta_squared(groups: list[np.ndarray]) -> float:
    """Fraction of a loading's total variance the grouping explains,
    ``SS_between / SS_total``.

    The effect size, not the p-value, is what the bootstrap carries: it is a
    variance ratio, so it does not move with a replicate's sample count the
    way a p-value does.
    """
    values = np.concatenate(groups)
    grand_mean = float(values.mean())
    ss_total = float(((values - grand_mean) ** 2).sum())
    ss_between = float(
        sum(group.size * (float(group.mean()) - grand_mean) ** 2 for group in groups)
    )
    return ss_between / ss_total if ss_total > 0 else np.nan


def _group_mean_diff(
    level_names: list[str], groups: list[np.ndarray], control_label: str | None
) -> float:
    """Signed ``mean(other) - mean(control)`` when there are exactly two
    levels; NaN otherwise.

    This is the only sign-bearing quantity the ANOVA reports, and so the one
    that carries direction -- which group loads higher -- through the
    bootstrap's sign correction, exactly as the logistic regression weight
    did. F, P and eta-squared are sign-invariant by construction.
    """
    if len(groups) != 2:
        return np.nan
    control_index = (
        level_names.index(control_label) if control_label in level_names else 0
    )
    control = groups[control_index]
    other = groups[1 - control_index]
    return float(other.mean() - control.mean())
