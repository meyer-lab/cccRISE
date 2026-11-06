"""The RISE condition factor (A; conditions ordered by their most heavily
loaded component, then by that loading), each component divided by its
largest magnitude (keeping sign)."""

from ... import common
from . import _common as S

AXES = ("wide",)  # as wide as `genes`' "tall", which it sits above


def plot(axs):
    draw(axs, S.DATASET, "BAL sample")


def draw(axs, dataset: str, sample_noun: str) -> None:
    """`dataset`'s RISE condition factor, its conditions called `sample_noun`
    in the source data."""
    (ax,) = axs
    factor = S.rise_factors(dataset)["pf2_A"]
    conditions = factor.values.astype(float)
    order = common.dominant_order(conditions)
    common.plot_factor_heatmap(ax, conditions[order], S.XLABEL)
    common.source_data(
        "rise_condition_factor",
        S.normalized_table(
            factor["condition"].values[order], conditions[order], "Sample"
        ),
        f"RISE condition factor (A): each {sample_noun}'s (row, top to bottom as "
        "plotted) loading on each RISE component, divided by the component's "
        "largest |loading| (sign kept).",
    )
    ax.set_xticks([])  # shares `genes`' component axis below it
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set_ylabel(f"Condition (n={len(conditions)})")


def makeFigure():
    return common.standalone_fixed(__name__)
