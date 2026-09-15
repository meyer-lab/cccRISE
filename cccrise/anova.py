"""One-way ANOVA of CP condition-factor loadings against a phenotype.

Replaces the L1 logistic regression that used to score `X.uns["A"]`.
The grouping is carried as a named label per condition rather than as an
integer code: `prepare_anova_inputs` collapses to two named levels when told
to, and otherwise passes every level of the group column through. Moving from
control-vs-ALAD to three-level `ALADstatus`, or to Died/CLAD/No CLAD, is then
an argument change rather than a code change.

Includes multiple testing correction support (Benjamini-Hochberg FDR,
Benjamini-Yekutieli, Bonferroni, Holm) and support for running ANOVA on saved
decomposition factors (e.g. `bal_alad_factors.nc`).
"""

from collections.abc import Sequence
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import f_oneway, false_discovery_control

from .figures.commonFuncs.plotGeneral import collapse_group, get_sample_to_group
from .utils import correct_conditions


def adjust_pvalues(
    p_values: Sequence[float] | np.ndarray,
    method: str | None = "fdr_bh",
) -> np.ndarray:
    """Adjust p-values for multiple hypothesis testing across components.

    Parameters
    ----------
    p_values : Sequence[float] | np.ndarray
        Raw p-values.
    method : str | None, default="fdr_bh"
        Correction method:
        - "fdr_bh" / "bh" / "fdr" / "benjamini_hochberg": Benjamini-Hochberg FDR
        - "fdr_by" / "by" / "benjamini_yekutieli": Benjamini-Yekutieli FDR
        - "bonferroni" / "bonf": Bonferroni family-wise error rate correction
        - "holm" / "holm_bonferroni": Holm-Bonferroni step-down correction
        - None / "none" / "raw": No correction (returns raw p-values)

    Returns
    -------
    np.ndarray
        Adjusted p-values, preserving NaNs for non-finite inputs.
    """
    ps = np.asarray(p_values, dtype=float)
    if method is None or str(method).lower() in ("none", "raw"):
        return ps.copy()

    method_key = str(method).lower().replace("-", "_")
    finite_mask = np.isfinite(ps)
    if not np.any(finite_mask):
        return ps.copy()

    finite_ps = ps[finite_mask]
    n_finite = len(finite_ps)

    if method_key in ("fdr_bh", "bh", "fdr", "benjamini_hochberg"):
        adjusted_finite = np.asarray(
            false_discovery_control(finite_ps, method="bh"), dtype=float
        )
    elif method_key in ("fdr_by", "by", "benjamini_yekutieli"):
        adjusted_finite = np.asarray(
            false_discovery_control(finite_ps, method="by"), dtype=float
        )
    elif method_key in ("bonferroni", "bonf"):
        adjusted_finite = np.clip(finite_ps * n_finite, 0.0, 1.0)
    elif method_key in ("holm", "holm_bonferroni"):
        order = np.argsort(finite_ps)
        sorted_ps = finite_ps[order]
        holm_adj = np.empty(n_finite, dtype=float)
        running_max = 0.0
        for i in range(n_finite):
            val = (n_finite - i) * sorted_ps[i]
            running_max = max(running_max, val)
            holm_adj[i] = min(running_max, 1.0)
        adjusted_finite = np.empty(n_finite, dtype=float)
        adjusted_finite[order] = holm_adj
    else:
        raise ValueError(
            f"Unknown multiple test correction method: {method!r}. "
            "Supported methods: 'fdr_bh', 'fdr_by', 'bonferroni', 'holm', 'none'."
        )

    out = np.full_like(ps, np.nan, dtype=float)
    out[finite_mask] = adjusted_finite
    return out


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
    correction: str | None = None,
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
        Threshold `Significant` is reported against.
    control_label : str | None, optional
        Level `GroupMeanDiff` is measured *from*. Defaults to the
        alphabetically first level, which is the level `binarize_group`'s
        codes made class 0, so the sign keeps the meaning it had under the
        logistic regression.
    correction : str | None, optional
        Multiple hypothesis testing correction across components (e.g., 'fdr_bh',
        'bonferroni', 'holm', or None). When specified, `Significant` is evaluated
        against the adjusted p-values in `PAdj`.

    Returns
    -------
    pd.DataFrame
        One row per component (1-indexed), columns:
        - 'Component': component index (1-based)
        - 'F': ANOVA F-statistic
        - 'P': raw p-value
        - 'PAdj': multiple-testing adjusted p-value (or raw p-value if correction is None)
        - 'EtaSquared': effect size eta^2 (SS_between / SS_total)
        - 'GroupMeanDiff': signed group mean difference (for binary grouping)
        - 'Significant': True if PAdj < alpha (or P < alpha when uncorrected)
        - 'NGroups': number of distinct groups with >=2 samples
        - 'GroupSizes': count of samples in each group
    """
    A = np.asarray(X.uns["A"], dtype=float)
    labels = pd.Series(np.asarray(sample_to_group), dtype=object).reset_index(drop=True)

    rows = []
    raw_p_values = []
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

        raw_p_values.append(p_value)
        rows.append(
            {
                "Component": component,
                "F": f_stat,
                "P": p_value,
                "EtaSquared": eta_squared,
                "GroupMeanDiff": group_mean_diff,
                "NGroups": len(groups),
                "GroupSizes": group_sizes,
            }
        )

    adj_p_values = adjust_pvalues(raw_p_values, method=correction)
    df = pd.DataFrame(rows)
    df["PAdj"] = adj_p_values
    df["Significant"] = [
        bool(padj < alpha) if np.isfinite(padj) else False for padj in adj_p_values
    ]

    cols = [
        "Component",
        "F",
        "P",
        "PAdj",
        "EtaSquared",
        "GroupMeanDiff",
        "Significant",
        "NGroups",
        "GroupSizes",
    ]
    return df[cols]


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


def run_saved_factors_anova(
    factors: str | Path | xr.Dataset | anndata.AnnData,
    adata: anndata.AnnData | None = None,
    sample_to_group: pd.Series | None = None,
    group_column: str | None = None,
    condition_column: str | None = None,
    control_label: str | None = None,
    other_label: str | None = None,
    alpha: float = 0.05,
    correction: str | None = "fdr_bh",
) -> pd.DataFrame:
    """Run one-way ANOVA on a saved factors dataset (e.g. bal_alad_factors.nc).

    Parameters
    ----------
    factors : str, Path, xr.Dataset, or anndata.AnnData
        Path to a saved NetCDF factor dataset, an xarray Dataset, or an AnnData
        holding decomposition factor matrices in `uns`.
    adata : anndata.AnnData, optional
        AnnData containing condition and group annotations in `obs`. If omitted
        and `factors` is a path or NetCDF dataset registered in
        `cccrise.runners.datasets.DATASETS` (e.g., 'alad' for 'bal_alad_factors.nc'),
        the dataset is automatically loaded.
    sample_to_group : pd.Series, optional
        Pre-extracted condition-to-group mapping Series.
    group_column : str, optional
        Phenotype group column in `adata.obs` (e.g. 'ALADstatus'). Defaults to
        the registered dataset's `group_column`.
    condition_column : str, optional
        Sample/condition column in `adata.obs`. Defaults to the registered
        dataset's `condition_column` or `factors_ds.attrs["condition_key"]`.
    control_label, other_label : str, optional
        When provided, collapses `group_column` into two binary levels.
        When omitted, tests every level of `group_column`.
    alpha : float, default=0.05
        Significance threshold for `Significant`.
    correction : str | None, default="fdr_bh"
        Multiple testing correction method: "fdr_bh", "bonferroni", "holm", or None.

    Returns
    -------
    pd.DataFrame
        ANOVA results table per component with F, P, PAdj, EtaSquared,
        GroupMeanDiff, and Significant flags.
    """
    from .factors import load_factors_netcdf, load_factors_to_anndata
    from .runners.datasets import DATASETS

    factors_ds: xr.Dataset | None = None
    if isinstance(factors, (str, Path)):
        factors_ds = load_factors_netcdf(factors)
    elif isinstance(factors, xr.Dataset):
        factors_ds = factors

    if factors_ds is not None:
        ds_name = str(
            factors_ds.attrs.get("dataset_name", factors_ds.attrs.get("dataset", ""))
        )
        if sample_to_group is None and adata is None and ds_name in DATASETS:
            spec = DATASETS[ds_name]
            try:
                adata = spec.load()
                group_column = group_column or spec.group_column
                condition_column = condition_column or spec.condition_column
            except Exception:
                try:
                    adata = spec.loader(filename=str(spec.source_file), normalize=False)
                    group_column = group_column or spec.group_column
                    condition_column = condition_column or spec.condition_column
                except Exception as err:
                    if sample_to_group is None:
                        raise RuntimeError(
                            f"Could not automatically load dataset {ds_name!r}: {err}. "
                            "Provide `adata` or `sample_to_group` directly."
                        ) from err

        cond_col = condition_column or str(
            factors_ds.attrs.get("condition_key", "sample")
        )
        factors_adata = load_factors_to_anndata(
            factors_ds, adata=adata, condition_column=cond_col
        )
    elif isinstance(factors, anndata.AnnData):
        factors_adata = factors
        cond_col = condition_column or "sample"
    else:
        raise TypeError(f"Unsupported factors type: {type(factors)}")

    if sample_to_group is None:
        if group_column is None:
            raise ValueError(
                "group_column must be specified if sample_to_group is not provided."
            )
        sample_to_group = get_sample_to_group(factors_adata, cond_col, group_column)

    if control_label is not None and other_label is not None:
        sample_to_group = collapse_group(
            sample_to_group, control_label, control_label, other_label
        )
    elif (control_label is None) ^ (other_label is None):
        raise ValueError(
            "Specify both control_label and other_label, or neither to test all levels."
        )

    return anova_component_associations(
        factors_adata,
        sample_to_group,
        alpha=alpha,
        control_label=control_label,
        correction=correction,
    )
