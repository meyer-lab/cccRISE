"""The IBD RISE condition factor (A), each component divided by its largest
magnitude (keeping sign), as the ALAD supplement draws ALAD's: biopsies
grouped by inflammation status (a color strip left of the matrix, each
status named at a bracket, as Figure 4A names them), and within a status
ordered by their most heavily loaded component, then by that loading."""

import numpy as np
import pandas as pd

from ... import common
from ...figure2.figureS_alad_cohort._rise_factors import XLABEL, normalized_table
from .. import _common as S
from ..cp_factor_matrices import STRIPS
from ._common import component_ticks

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    factor = S.factors()["pf2_A"]
    conditions = factor.values.astype(float)
    names = factor["condition"].values.astype(str)
    status = (
        S.samples().set_index(S.SPEC.condition_column)[S.GROUP_COLUMN].loc[names]
    ).to_numpy()
    rank = np.array([list(S.GROUP_LABELS).index(s) for s in status])
    order = common.dominant_order(conditions)
    order = order[np.argsort(rank[order], kind="stable")]
    common.plot_factor_heatmap(ax, conditions[order], XLABEL)
    component_ticks(ax, conditions.shape[1])
    common.source_data(
        "rise_condition_factor",
        normalized_table(names[order], conditions[order], "Sample").assign(
            Status=status[order]
        ),
        "RISE condition factor (A): each terminal ileum biopsy's (row, top to "
        "bottom as plotted) loading on each RISE component, divided by the "
        "component's largest |loading| (sign kept), with its inflammation status.",
    )
    ax.set_yticks([])
    ax.set_ylabel(f"Conditions (n={len(conditions)})")
    common.condition_strips(
        ax, pd.DataFrame({S.GROUP_COLUMN: status[order]}), STRIPS, brackets=True
    )


def makeFigure():
    return common.standalone_fixed(__name__)
