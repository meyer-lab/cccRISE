"""The dataset Figure 5 and its supplements draw -- the IBDverse terminal
ileum cohort (the registry's `ibd` entry: RISE 60 / CP 6, mesenchymal lineage
excluded, at most 2,000 cells per biopsy) -- the `cccrise-cohort` tables that
summarize its biopsies and cell types, and the inflammation-status
associations tested against its components, as Figure 3 tests ALAD's."""

from functools import cache
from typing import Any

import anndata
import numpy as np
import pandas as pd
import xarray as xr

from ...anova import run_multiple_factors_anova
from ...factors import load_factors_netcdf
from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...utils import correct_by_read_depth
from .. import common

DATASET = "ibd"
SPEC = DATASETS[DATASET]
GROUP_COLUMN = "inflammation_status"

#: `inflammation_status` levels, in disease order, and how figures name them
GROUP_LABELS = {
    "Healthy": "Healthy",
    "CD_uninflamed": "CD uninflamed",
    "CD_inflamed": "CD inflamed",
}
GROUP_COLORS = dict(zip(GROUP_LABELS, common.style.CATEGORICAL, strict=False))
_UNINFLAMED, _INFLAMED = list(GROUP_LABELS)[1:]

#: Inflammation-status associations tested against every component, as
#: `figure3._common.ASSOCIATIONS` (see `anova.prepare_group_labels`): the
#: three-group test and its pairwise contrasts
ASSOCIATIONS: list[dict[str, Any]] = [
    {"name": "Inflammation status (3-way)", "group_column": GROUP_COLUMN},
    {
        "name": "Healthy vs. CD",
        "group_column": GROUP_COLUMN,
        "control_label": "Healthy",
        "other_label": "CD",
    },
    {
        "name": "Healthy vs. CD uninflamed",
        "group_column": GROUP_COLUMN,
        "levels": ["Healthy", _UNINFLAMED],
        "control_label": "Healthy",
    },
    {
        "name": "Healthy vs. CD inflamed",
        "group_column": GROUP_COLUMN,
        "levels": ["Healthy", _INFLAMED],
        "control_label": "Healthy",
    },
    {
        "name": "CD uninflamed vs. inflamed",
        "group_column": GROUP_COLUMN,
        "levels": [_UNINFLAMED, _INFLAMED],
        "control_label": _UNINFLAMED,
    },
]
THREE_WAY = ASSOCIATIONS[0]["name"]
#: The pairwise contrasts' levels, by association name
CONTRASTS = {a["name"]: a["levels"] for a in ASSOCIATIONS if "levels" in a}


def _load(table: str) -> pd.DataFrame:
    return common.load_runner_output(
        f"{table}_{SPEC.output_key}.parquet",
        VALIDATION_DIR,
        hint=f"    uv run cccrise-cohort --dataset {DATASET}",
    )


@cache
def samples() -> pd.DataFrame:
    """One row per biopsy, in the fit's condition order: its sample-level
    metadata, cell count, read depth, and raw and read-depth-corrected CP
    loadings (`run_cohort.sample_table`)."""
    return _load("cohort_samples")


@cache
def celltype_counts() -> pd.DataFrame:
    """Cells per biopsy x lineage x fine cell type."""
    return _load("cohort_celltype_counts")


@cache
def celltype_weights() -> pd.DataFrame:
    """Mean per-cell sender and receiver weight of each lineage and fine cell
    type on every CP component (`run_cohort.celltype_weights`)."""
    return _load("celltype_component_weights")


@cache
def carrier_expression() -> pd.DataFrame:
    """Expression of each component's top LR pair genes per biopsy x fine
    cell type (`run_cohort.carrier_expression`)."""
    return _load("carrier_expression")


@cache
def factors() -> xr.Dataset:
    return load_factors_netcdf(SPEC.factors_path())


def corrected_loadings(factor: str) -> pd.DataFrame:
    """Condition factor `factor` (`cp_A` or `pf2_A`), read-depth corrected
    as the ANOVA tests it (`utils.correct_by_read_depth`), one row per
    biopsy in `samples` order and one column per component (1-indexed)."""
    A = factors()[factor]
    assert np.array_equal(A["condition"].astype(str), samples()[SPEC.condition_column])
    values = correct_by_read_depth(A.values.astype(float), samples()["read_depth"])
    return pd.DataFrame(values, columns=range(1, values.shape[1] + 1))


def condition_adata(loadings: pd.DataFrame) -> anndata.AnnData:
    """One observation per biopsy (`samples` order) carrying `loadings` as
    its condition factor (`uns["A"]`): what `anova` and
    `common.plot_category_component_weights` need, without the cells."""
    obs = samples().drop(columns=samples().filter(like="loading").columns)
    obs.index = obs[SPEC.condition_column].astype(str).to_numpy()
    return anndata.AnnData(obs=obs, uns={"A": loadings.to_numpy()})


def _association_table(factor: str) -> pd.DataFrame:
    return run_multiple_factors_anova(
        condition_adata(corrected_loadings(factor)),
        ASSOCIATIONS,
        condition_column=SPEC.condition_column,
        correction="fdr_bh",
        global_correction=True,
        read_depth_correct=False,
    )


@cache
def association_table() -> pd.DataFrame:
    """One-way ANOVA of every CP component's read-depth-corrected loading
    against every association, FDR-corrected across all associations and
    components (within cccRISE's CP fit only)."""
    return _association_table("cp_A")


@cache
def rise_association_table() -> pd.DataFrame:
    """As `association_table`, for every RISE component's condition factor
    (`pf2_A`)."""
    return _association_table("pf2_A")
