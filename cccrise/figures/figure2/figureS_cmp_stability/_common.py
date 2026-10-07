"""The per-component jackknife stability box plot shared by this figure's RISE
and CP panels, and the bimodality rule its histogram panel selects components
by, from ``uv run cccrise-stability``.

Only the jackknife is shown: seed refits reproduce every component at FMS
~1.0, so they appear only in Figure 2's aggregate panel.
"""

from functools import cache

import numpy as np
import pandas as pd
import seaborn as sns

from ....runners.datasets import DATASETS, VALIDATION_DIR
from ....validation.stability import JACKKNIFE_AXIS
from ... import common, style

DATASET = "alad"
COMPONENT_NAMES = {"RISE": "RISE", "CPD": "CP"}  # stage -> axis label
COLOR = style.STABILITY_AXIS_COLORS[JACKKNIFE_AXIS]
#: A component's replicate FMS is bimodal when its sorted values have a gap of
#: at least `BIMODAL_GAP` with at least `BIMODAL_MINOR_FRACTION` of the
#: replicates on the smaller side -- a second population rather than one or
#: two replicates that failed to re-identify the component.
BIMODAL_GAP = 0.25
BIMODAL_MINOR_FRACTION = 0.1


@cache
def stability() -> pd.DataFrame:
    """Jackknife rows, one per (replicate, stage, component)."""
    df = common.load_runner_output(
        DATASETS[DATASET].stability_filename(), VALIDATION_DIR
    )
    return df[df["Axis"] == JACKKNIFE_AXIS]


def is_bimodal(fms: np.ndarray) -> bool:
    """Whether `fms` splits into two populations at its largest gap; see
    `BIMODAL_GAP` and `BIMODAL_MINOR_FRACTION`."""
    fms = np.sort(np.asarray(fms, dtype=float))
    if fms.size < 2:
        return False
    split = int(np.argmax(np.diff(fms)))
    minor = min(split + 1, fms.size - split - 1)
    return bool(
        fms[split + 1] - fms[split] >= BIMODAL_GAP
        and minor >= BIMODAL_MINOR_FRACTION * fms.size
    )


def bimodal_components(df: pd.DataFrame) -> list[tuple[str, int]]:
    """(stage, component) pairs, RISE first, whose replicate FMS is bimodal
    (`is_bimodal`)."""
    flagged = df.groupby(["Stage", "Component"])["FMS"].agg(
        lambda fms: is_bimodal(fms.to_numpy())
    )
    return [
        (stage, int(component))
        for stage in COMPONENT_NAMES
        for (s, component), bimodal in sorted(flagged.items())
        if bimodal and s == stage
    ]


def plot_stage(fig, stage: str) -> None:
    """Per-component jackknife FMS for `stage` ('RISE' or 'CPD'), one box per
    component."""
    ax = fig.subplots()
    sns.boxplot(
        data=stability()[stability()["Stage"] == stage],
        x="Component",
        y="FMS",
        color=COLOR,
        fliersize=1,
        linewidth=style.LINE_WIDTH_THIN,
        ax=ax,
    )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{COMPONENT_NAMES[stage]} component")
    ax.set_ylabel("Jackknife factor match score")
