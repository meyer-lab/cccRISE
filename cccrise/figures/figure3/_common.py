"""Data shared by Figure 3's panels: the ALAD outcome associations tested
against every CP component and every RISE component, and the fit they are
tested on.

`ASSOCIATIONS` is also the Shiny app's list of ALAD comparisons.
"""

import copy
from functools import cache
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ...anova import _resolve_factors_and_metadata, run_multiple_factors_anova
from ...factors import load_factors_netcdf
from ...runners.datasets import DATASETS
from .. import common, style

DATASET = "alad"
SPEC = DATASETS[DATASET]

# The CP component associated with the 6-month outcome
COMPONENT = 4
OUTCOME = "6-Month Outcome (4-Way)"

#: Outcome associations tested against every CP component: `name` labels the
#: row, `group_column` is the condition-level `obs` column, and the optional
#: `levels` / `mapping` / `control_label` / `other_label` regroup it (see
#: `anova.prepare_group_labels`).
ASSOCIATIONS: list[dict[str, Any]] = [
    {
        "name": "Control vs. ALAD (All)",
        "group_column": "ALADstatus",
        "control_label": "control",
        "other_label": "alad",
    },
    {
        "name": "ALAD Status (3-Way)",
        "group_column": "ALADstatus",
    },
    {
        "name": "6-Month Outcome (4-Way)",
        "group_column": "6monthcondition",
    },
    # CLAD among survivors; death against every survivor
    {
        "name": "6-Month CLAD vs. Stable/Recovered",
        "group_column": "6monthcondition",
        "mapping": {
            "CLAD": "CLAD",
            "No CLAD/Stable": "Stable/Recovered",
            "No CLAD/Recovered": "Stable/Recovered",
        },
        "control_label": "Stable/Recovered",
    },
    {
        "name": "6-Month Died vs. Alive",
        "group_column": "6monthcondition",
        "mapping": {
            "Died": "Died",
            "CLAD": "Alive",
            "No CLAD/Stable": "Alive",
            "No CLAD/Recovered": "Alive",
        },
        "control_label": "Alive",
    },
    # The one No CLAD/Declined sample fits none of the four groups and is left out
    {
        "name": "1-Year Outcome (4-Way)",
        "group_column": "1yearcondition",
        "mapping": {
            "No CLAD/Stable": "No CLAD/Stable",
            "No CLAD/Recovered": "No CLAD/Recovered",
            "Recovered": "No CLAD/Recovered",
            "CLAD/Declined": "CLAD",
            "CLAD/Stable": "CLAD",
            "Died": "Died",
        },
    },
    {
        "name": "1-Year CLAD vs. No CLAD",
        "group_column": "1yearcondition",
        "mapping": {
            "CLAD/Declined": "CLAD",
            "CLAD/Stable": "CLAD",
            "No CLAD/Stable": "No CLAD",
            "No CLAD/Recovered": "No CLAD",
            "No CLAD/Declined": "No CLAD",
            "Recovered": "No CLAD",
        },
        "control_label": "No CLAD",
    },
    {
        "name": "1-Year Died vs. Alive",
        "group_column": "1yearcondition",
        "mapping": {
            "Died": "Died",
            "CLAD/Declined": "Alive",
            "CLAD/Stable": "Alive",
            "No CLAD/Stable": "Alive",
            "No CLAD/Recovered": "Alive",
            "No CLAD/Declined": "Alive",
            "Recovered": "Alive",
        },
        "control_label": "Alive",
    },
    {
        "name": "Acute Rejection (Yes vs. No)",
        "group_column": "rejection.y",
        "levels": ["no", "yes"],
        "control_label": "no",
    },
]


@cache
def factors_adata() -> anndata.AnnData:
    """The canonical fit attached to the ALAD source data's annotations."""
    adata, _, _ = _resolve_factors_and_metadata(
        SPEC.factors_path(), condition_column=SPEC.condition_column
    )
    return adata


@cache
def association_table() -> pd.DataFrame:
    """ANOVA of every CP component against every association, FDR-corrected
    across all associations and components."""
    return run_multiple_factors_anova(
        factors_adata(),
        ASSOCIATIONS,
        condition_column=SPEC.condition_column,
        correction="fdr_bh",
        global_correction=True,
    )


def association(name: str) -> dict[str, Any]:
    return next(spec for spec in ASSOCIATIONS if spec["name"] == name)


@cache
def condition_obs() -> pd.DataFrame:
    """One obs row per condition, in the `np.unique` order
    `get_sample_to_group` gives labels in."""
    obs = cast(pd.DataFrame, factors_adata().obs)
    obs = obs.drop_duplicates(SPEC.condition_column)
    order = np.unique(obs[SPEC.condition_column])
    return obs.set_index(obs[SPEC.condition_column].to_numpy()).loc[order]


@cache
def rise_association_table() -> pd.DataFrame:
    """ANOVA of every RISE component's condition factor (`pf2_A`) against
    every association, as `association_table` tests the CP components,
    FDR-corrected across all associations and components."""
    pf2_A = load_factors_netcdf(SPEC.factors_path())["pf2_A"]
    # `get_sample_to_group` orders labels by `np.unique`, so pf2_A rows must too
    conditions = np.unique(factors_adata().obs[SPEC.condition_column]).astype(str)
    assert np.array_equal(conditions, pf2_A.condition.astype(str))
    adata = copy.copy(factors_adata())
    adata.uns = {**adata.uns, "A": pf2_A.values}
    return run_multiple_factors_anova(
        adata,
        ASSOCIATIONS,
        condition_column=SPEC.condition_column,
        correction="fdr_bh",
        global_correction=True,
    )


def plot_ranked_loadings(
    ax: plt.Axes,
    loadings: np.ndarray,
    n_marked: int,
    point_size: float = style.POINT_SIZE_BACKGROUND,
) -> None:
    """One component's loadings against their rank (largest first): the
    S-shaped curve with a steep head (the Shiny explorer's rank curve), the
    `n_marked` largest marked."""
    order = np.argsort(-loadings)
    ranks = np.arange(1, len(order) + 1)
    ax.plot(ranks, loadings[order], color=style.COLORS["ink"], lw=style.LINE_WIDTH)
    common.reference_line(ax, 0)
    ax.scatter(
        ranks[:n_marked],
        loadings[order[:n_marked]],
        color=style.COLORS["emphasis"],
        s=point_size,
        linewidths=0,
        zorder=3,
        clip_on=False,
    )
    low, high = float(loadings.min()), float(loadings.max())
    ax.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
    ax.set_xlim(0, len(order) + 1)
