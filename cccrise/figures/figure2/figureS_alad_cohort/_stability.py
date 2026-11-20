"""The per-component stability box plot shared by this figure's RISE and CP
panels, from ``uv run cccrise-stability``: each component's jackknife FMS,
then the aggregate FMS (a replicate's mean over components) under the
jackknife and seed refits.

Per component only the jackknife is shown: seed refits reproduce every
component at FMS ~1.0, so they appear only in the aggregate columns.
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


def plot_stage(ax, stage: str) -> None:
    """Jackknife FMS for `stage` ('RISE' or 'CPD'), one box per component,
    then one box per stability axis of the replicates' aggregate FMS, past a
    solid divider (dashed lines mark selected ranks elsewhere)."""
    rows = stability()[stability()["Stage"] == stage]
    name = COMPONENT_NAMES[stage]
    jackknife = rows[rows["Axis"] == JACKKNIFE_AXIS]
    common.source_data(
        f"jackknife_fms_{name.lower()}_components",
        replicate_table(jackknife),
        f"One row per leave-one-patient-out jackknife replicate and {name} "
        "component: the factor match score (FMS) of the replicate's "
        "best-matching component to the full-cohort component.",
    )
    aggregate = rows.groupby(["Axis", "Replicate"], as_index=False)["FMS"].mean()
    common.source_data(
        f"aggregate_fms_{name.lower()}_by_replicate",
        aggregate.rename(columns={"FMS": "Aggregate FMS"}),
        f"Each stability replicate's aggregate FMS (mean factor match score to "
        f"the full-cohort fit over {name} components), per stability axis "
        "(jackknife: leave one patient out; seed_only: refit from another seed).",
    )
    labels = [style.STABILITY_AXIS_LABELS[a] for a in AGGREGATE_AXES]
    columns = pd.concat(
        [
            jackknife.assign(Column=jackknife["Component"].astype(str)),
            aggregate.assign(Column=aggregate["Axis"].map(style.STABILITY_AXIS_LABELS)),
        ]
    )
    components = [str(c) for c in sorted(jackknife["Component"].unique())]
    sns.boxplot(
        data=columns,
        x="Column",
        y="FMS",
        hue="Axis",
        order=[*components, "", *labels],  # "" leaves a gap before the aggregates
        palette=style.STABILITY_AXIS_COLORS,
        dodge=False,
        legend=False,
        fliersize=1,
        linewidth=style.LINE_WIDTH_THIN,
        ax=ax,
    )
    ax.axvline(len(components), **style.SPACER)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels())
    for tick in ax.get_xticklabels()[-len(labels) :]:
        tick.set(rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{name} component")
    ax.set_ylabel("Factor match score")
