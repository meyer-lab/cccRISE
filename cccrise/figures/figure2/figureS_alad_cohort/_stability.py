"""Stability of the ALAD fit from ``uv run cccrise-stability``: the
per-component jackknife box plot shared by this figure's RISE and CPD panels,
and the aggregate FMS (a replicate's mean over components) Figure 2 plots
under the jackknife and seed refits.

Per component only the jackknife is shown: seed refits reproduce every
component at FMS ~1.0, so they appear only in the aggregate.
"""

from functools import cache

import pandas as pd
import seaborn as sns

from ....runners.datasets import DATASETS, VALIDATION_DIR
from ....validation.stability import JACKKNIFE_AXIS
from ... import common, style

DATASET = "alad"
COMPONENT_NAMES = {"RISE": "RISE", "CPD": "CPD"}  # stage -> axis label
AGGREGATE_AXES = (JACKKNIFE_AXIS, "seed_only")


@cache
def stability() -> pd.DataFrame:
    """Rows of the plotted stability axes, one per (axis, replicate, stage,
    component)."""
    df = common.load_runner_output(
        DATASETS[DATASET].stability_filename(), VALIDATION_DIR
    )
    return df[df["Axis"].isin(AGGREGATE_AXES)]


def replicate_table(rows: pd.DataFrame) -> pd.DataFrame:
    """`stability()` rows as source data: the replicate, its left-out
    sample, and each component's match."""
    return rows[
        [
            "Replicate",
            "LeftOut",
            "Seed",
            "Stage",
            "Component",
            "MatchedComponent",
            "FMS",
        ]
    ].rename(
        columns={
            "LeftOut": "Left-out sample",
            "Stage": "Model",
            "MatchedComponent": "Matched replicate component",
        }
    )


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Each replicate's aggregate FMS: its mean over components."""
    return rows.groupby(["Stage", "Axis", "Replicate"], as_index=False)["FMS"].mean()


def plot_stage(ax, stage: str) -> None:
    """Jackknife FMS for `stage` ('RISE' or 'CPD'), one box per component."""
    rows = stability()
    jackknife = rows[(rows["Stage"] == stage) & (rows["Axis"] == JACKKNIFE_AXIS)]
    name = COMPONENT_NAMES[stage]
    common.source_data(
        f"jackknife_fms_{name.lower()}_components",
        replicate_table(jackknife),
        f"One row per leave-one-patient-out jackknife replicate and {name} "
        "component: the factor match score (FMS) of the replicate's "
        "best-matching component to the full-cohort component.",
    )
    sns.boxplot(
        data=jackknife,
        x="Component",
        y="FMS",
        color=style.STABILITY_AXIS_COLORS[JACKKNIFE_AXIS],
        fliersize=1,
        linewidth=style.LINE_WIDTH_THIN,
        ax=ax,
    )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{name} component")
    ax.set_ylabel("FMS")
