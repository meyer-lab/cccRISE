"""One-way ANOVA of CP condition-factor loadings against patient status variables.

Replaces the L1 logistic regression that used to score `X.uns["A"]`.
Supports multi-level categorical ANOVA, binary collapsed comparisons, subset level
contrasts (e.g., Control vs. Declined, Control vs. Recovered, Declined vs. Recovered),
and arbitrary recodings across any patient status column in `obs` (e.g. `ALADstatus`,
`6monthcondition`, `1yearcondition`, `diagnosisgroup`, `cmvstatus`).

Includes multiple testing correction support (Benjamini-Hochberg FDR,
Benjamini-Yekutieli, Bonferroni, Holm) and batch execution of multiple association
specifications on saved decomposition factors (e.g. `bal_alad_factors.nc`).
"""

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import f_oneway, false_discovery_control

from .utils import correct_conditions


def get_sample_to_group(
    X: anndata.AnnData, condition_column: str, group_col: str
) -> pd.Series:
    """Map each unique sample to its group label, ordered to match
    np.unique(X.obs[condition_column])."""
    sample_to_group = (
        cast(pd.DataFrame, X.obs)
        .drop_duplicates(subset=[condition_column, group_col])
        .set_index(condition_column)[group_col]
    )
    return sample_to_group.loc[np.unique(X.obs[condition_column], return_index=True)[0]]


def collapse_group(
    sample_to_group: pd.Series, match_value: str, match_name: str, other_name: str
) -> pd.Series:
    """Collapse a group Series into two named categories, keeping the names.

    The names, rather than codes, are what a test that reports per-group
    statistics needs -- an ANOVA has to be able to say which group loaded
    higher. `binarize_group` is this with the names traded for codes.
    """
    return sample_to_group.map(lambda x: match_name if x == match_value else other_name)


def binarize_group(
    sample_to_group: pd.Series, match_value: str, match_name: str, other_name: str
) -> pd.Series:
    """Collapse a group Series into two named categories and return integer codes."""
    collapsed = collapse_group(sample_to_group, match_value, match_name, other_name)
    return collapsed.astype("category").cat.codes


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


def prepare_group_labels(
    sample_to_group: pd.Series,
    levels: Sequence[str] | None = None,
    mapping: Mapping[str, str] | None = None,
    control_label: str | None = None,
    other_label: str | None = None,
    condition_order: Sequence[str] | None = None,
) -> pd.Series:
    """Filter, recode, or collapse a condition-to-group mapping Series.

    Parameters
    ----------
    sample_to_group : pd.Series
        Raw mapping from sample/condition identifiers to phenotype labels.
    levels : Sequence[str] | None, optional
        Subset of group levels to retain. Conditions with other levels are set to NaN.
    mapping : Mapping[str, str] | None, optional
        Dictionary mapping original group labels to new categories.
    control_label, other_label : str | None, optional
        When both are provided, collapses groups into binary levels:
        `control_label` is retained, and all other non-null entries become `other_label`.
    condition_order : Sequence[str] | None, optional
        Order of conditions to align the index against.

    Returns
    -------
    pd.Series
        Processed group labels aligned to conditions.
    """
    s = sample_to_group.copy()
    if condition_order is not None and s.index.size > 0:
        s = s.reindex(condition_order)

    if mapping is not None:
        s = s.map(mapping)

    if levels is not None:
        level_set = set(levels)
        s = s.where(s.isin(level_set))

    if control_label is not None and other_label is not None:
        s = collapse_group(s, control_label, control_label, other_label)
    elif control_label is None and other_label is not None:
        raise ValueError(
            "Pass both control_label and other_label to collapse to binary levels."
        )

    return s


def prepare_anova_inputs(
    adata: anndata.AnnData,
    factors: list[np.ndarray],
    condition_column: str,
    group_column: str,
    control_label: str | None = None,
    other_label: str | None = None,
    levels: Sequence[str] | None = None,
    mapping: Mapping[str, str] | None = None,
) -> tuple[anndata.AnnData, pd.Series]:
    """Attach one CP fit's factors to an AnnData object and derive the group
    label the ANOVA is run against.

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
        levels.
    levels : Sequence[str] | None, optional
        Subset of group levels to retain.
    mapping : Mapping[str, str] | None, optional
        Dictionary mapping original group labels to new categories.

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
    processed_labels = prepare_group_labels(
        sample_to_group,
        levels=levels,
        mapping=mapping,
        control_label=control_label,
        other_label=other_label,
    )
    return X, processed_labels


def anova_component_associations(
    X: anndata.AnnData,
    sample_to_group: pd.Series,
    alpha: float = 0.05,
    control_label: str | None = None,
    correction: str | None = None,
    min_group_size: int = 2,
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
    min_group_size : int, default=2
        A component's test is reported as NaN when any group has fewer
        conditions than this. 1 keeps a single-condition group in the test
        (it adds between-group but no within-group degrees of freedom).

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
        - 'NGroups': number of distinct groups with a sample
        - 'GroupSizes': count of samples in each group
    """
    A = np.asarray(X.uns["A"], dtype=float)
    labels = pd.Series(np.asarray(sample_to_group), dtype=object).reset_index(drop=True)
    if len(labels) != A.shape[0]:
        raise ValueError(
            f"Length of sample_to_group ({len(labels)}) does not match condition "
            f"factor rows in X.uns['A'] ({A.shape[0]})."
        )

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

        if len(groups) < 2 or any(size < min_group_size for size in group_sizes):
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

    The effect size, not the p-value, is what the stability replicates carry: it is a
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
    stability replicates' sign correction, exactly as the logistic regression weight
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


def _resolve_factors_and_metadata(
    factors: str | Path | xr.Dataset | anndata.AnnData,
    adata: anndata.AnnData | None = None,
    condition_column: str | None = None,
    group_column: str | None = None,
) -> tuple[anndata.AnnData, str, str]:
    """Resolve factor matrices and AnnData metadata annotations."""
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
        if adata is None and ds_name in DATASETS:
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
                    raise RuntimeError(
                        f"Could not automatically load dataset {ds_name!r}: {err}. "
                        "Provide `adata` directly."
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

    grp_col = group_column or (
        "ALADstatus" if "ALADstatus" in factors_adata.obs.columns else ""
    )
    return factors_adata, cond_col, grp_col


def read_depth_corrected(adata: anndata.AnnData) -> anndata.AnnData:
    """A shallow copy of `adata` with `uns["A"]` read-depth corrected
    (`utils.correct_conditions`)."""
    corrected = copy.copy(adata)
    corrected.uns = {**adata.uns, "A": correct_conditions(adata)}
    return corrected


def run_saved_factors_anova(
    factors: str | Path | xr.Dataset | anndata.AnnData,
    adata: anndata.AnnData | None = None,
    sample_to_group: pd.Series | None = None,
    group_column: str | None = None,
    condition_column: str | None = None,
    control_label: str | None = None,
    other_label: str | None = None,
    levels: Sequence[str] | None = None,
    mapping: Mapping[str, str] | None = None,
    alpha: float = 0.05,
    correction: str | None = "fdr_bh",
    association_name: str | None = None,
    read_depth_correct: bool = True,
) -> pd.DataFrame:
    """Run one-way ANOVA on a saved factors dataset (e.g. bal_alad_factors.nc).

    Parameters
    ----------
    factors : str, Path, xr.Dataset, or anndata.AnnData
        Path to a saved NetCDF factor dataset, an xarray Dataset, or an AnnData
        holding decomposition factor matrices in `uns`.
    adata : anndata.AnnData, optional
        AnnData containing condition and group annotations in `obs`.
    sample_to_group : pd.Series, optional
        Pre-extracted condition-to-group mapping Series.
    group_column : str, optional
        Phenotype group column in `adata.obs`.
    condition_column : str, optional
        Sample/condition column in `adata.obs`.
    control_label, other_label : str, optional
        When provided, collapses `group_column` into two binary levels.
    levels : Sequence[str], optional
        Subset of group levels to retain (e.g. ['control', 'declined']).
    mapping : Mapping[str, str], optional
        Dictionary mapping original group levels to new categories.
    alpha : float, default=0.05
        Significance threshold for `Significant`.
    correction : str | None, default="fdr_bh"
        Multiple testing correction method: "fdr_bh", "bonferroni", "holm", or None.
    association_name : str, optional
        Optional name to record in an 'Association' column.
    read_depth_correct : bool, default=True
        Test the read-depth-corrected condition loadings
        (`utils.correct_conditions`), as `prepare_anova_inputs` does; needs
        the single-cell data in `X`. False tests `uns["A"]` as is, for
        matrices that are not component loadings.

    Returns
    -------
    pd.DataFrame
        ANOVA results table per component with F, P, PAdj, EtaSquared,
        GroupMeanDiff, and Significant flags.
    """
    factors_adata, cond_col, grp_col = _resolve_factors_and_metadata(
        factors,
        adata=adata,
        condition_column=condition_column,
        group_column=group_column,
    )
    if read_depth_correct:
        factors_adata = read_depth_corrected(factors_adata)

    if sample_to_group is None:
        target_grp_col = group_column or grp_col
        if not target_grp_col:
            raise ValueError(
                "group_column must be specified if sample_to_group is not provided."
            )
        sample_to_group = get_sample_to_group(factors_adata, cond_col, target_grp_col)

    processed_labels = prepare_group_labels(
        sample_to_group,
        levels=levels,
        mapping=mapping,
        control_label=control_label,
        other_label=other_label,
    )

    df = anova_component_associations(
        factors_adata,
        processed_labels,
        alpha=alpha,
        control_label=control_label,
        correction=correction,
    )
    if association_name is not None:
        df["Association"] = association_name
        df["GroupColumn"] = group_column or grp_col
    return df


def run_multiple_factors_anova(
    factors: str | Path | xr.Dataset | anndata.AnnData,
    associations: Sequence[dict[str, Any] | str],
    adata: anndata.AnnData | None = None,
    condition_column: str | None = None,
    alpha: float = 0.05,
    correction: str | None = "fdr_bh",
    global_correction: bool = False,
    read_depth_correct: bool = True,
) -> pd.DataFrame:
    """Run multiple ANOVA phenotype association tests on a saved factors dataset.

    Parameters
    ----------
    factors : str, Path, xr.Dataset, or anndata.AnnData
        Saved factor dataset or AnnData.
    associations : Sequence[dict[str, Any] | str]
        List of association configurations. Can be:
        - Dictionaries with keys like 'name', 'group_column', 'levels', 'mapping',
          'control_label', 'other_label', 'alpha', 'correction'
        - Column names (strings) in `adata.obs` to test across all their levels
    adata : anndata.AnnData, optional
        AnnData containing condition annotations in `obs`.
    condition_column : str, optional
        Condition column name.
    alpha : float, default=0.05
        Default significance threshold for configs that do not specify one.
    correction : str | None, default="fdr_bh"
        Default multiple test correction for configs that do not specify one.
    global_correction : bool, default=False
        If True, applies multiple hypothesis testing correction across all
        associations and components simultaneously.
    read_depth_correct : bool, default=True
        See `run_saved_factors_anova`.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing all ANOVA results with an added 'Association' column.
    """
    factors_adata, cond_col, _ = _resolve_factors_and_metadata(
        factors,
        adata=adata,
        condition_column=condition_column,
    )
    if read_depth_correct:
        factors_adata = read_depth_corrected(factors_adata)

    dfs: list[pd.DataFrame] = []
    for item in associations:
        if isinstance(item, str):
            assoc_dict: dict[str, Any] = {
                "name": item,
                "group_column": item,
                "alpha": alpha,
                "correction": correction,
            }
        elif isinstance(item, dict):
            assoc_dict = {
                "name": item.get("name", item["group_column"]),
                "group_column": item["group_column"],
                "levels": item.get("levels"),
                "mapping": item.get("mapping"),
                "control_label": item.get("control_label"),
                "other_label": item.get("other_label"),
                "alpha": item.get("alpha", alpha),
                "correction": item.get("correction", correction),
            }
        else:
            raise TypeError(f"Unsupported association item type: {type(item)}")

        sample_to_group = get_sample_to_group(
            factors_adata, cond_col, assoc_dict["group_column"]
        )
        processed_labels = prepare_group_labels(
            sample_to_group,
            levels=assoc_dict.get("levels"),
            mapping=assoc_dict.get("mapping"),
            control_label=assoc_dict.get("control_label"),
            other_label=assoc_dict.get("other_label"),
        )
        corr_to_use = None if global_correction else assoc_dict.get("correction")
        res = anova_component_associations(
            factors_adata,
            processed_labels,
            alpha=float(assoc_dict["alpha"])
            if assoc_dict.get("alpha") is not None
            else alpha,
            control_label=assoc_dict.get("control_label"),
            correction=corr_to_use,
        )
        res["Association"] = assoc_dict["name"]
        res["GroupColumn"] = assoc_dict["group_column"]
        dfs.append(res)

    combined_df = pd.concat(dfs, ignore_index=True)

    if global_correction:
        global_adj = adjust_pvalues(combined_df["P"].to_numpy(), method=correction)
        combined_df["PAdj"] = global_adj
        combined_df["Significant"] = [
            bool(padj < alpha) if np.isfinite(padj) else False for padj in global_adj
        ]

    return combined_df
