"""The RISE condition factor (A), each component divided by its largest
magnitude (keeping sign); conditions grouped by ALAD status, then six-month
outcome (`common.condition_strips`), and within a group ordered by their most
heavily loaded component, then by that loading."""

from ... import common
from ...figure3._common import condition_obs
from . import _common as S

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    factor = S.rise_factors()["pf2_A"]
    conditions = factor.values.astype(float)
    groups = condition_obs().loc[factor["condition"].values.astype(str)]
    order = common.group_conditions(groups, common.dominant_order(conditions))
    common.plot_factor_heatmap(ax, conditions[order], S.XLABEL)
    S.odd_component_ticks(ax, conditions.shape[1])
    common.source_data(
        "rise_condition_factor",
        S.normalized_table(
            factor["condition"].values[order], conditions[order], "Sample"
        ),
        "RISE condition factor (A): each BAL sample's (row, top to bottom as "
        "plotted) loading on each RISE component, divided by the component's "
        "largest |loading| (sign kept).",
    )
    ax.set_yticks([])
    ax.set_ylabel(f"Conditions (n={len(conditions)})")
    common.condition_strips(ax, groups.iloc[order])


def makeFigure():
    return common.standalone_fixed(__name__)
