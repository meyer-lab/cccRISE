"""Condition loadings of the CP component associated with the 6-month
outcome, per outcome level, with a patient's paired samples joined."""

from .. import common, style
from . import _common as S

REGROUPING_KEYS = ("levels", "mapping", "control_label", "other_label")
AXES = ("rect",)


def plot(axs):
    plot_outcome_weights(axs[0], S.OUTCOME, S.COMPONENT)


def plot_outcome_weights(ax, association: str, component: int) -> None:
    """CP `component`'s condition loadings per level of `association`, with
    its ANOVA statistics from `S.association_table`. Sample counts are left
    to the caption."""
    # TODO: plots read-depth-corrected loadings, but `S.association_table`
    # tests the uncorrected cp_A, so the boxes are not what was tested.
    spec = S.association(association)
    table = S.association_table()
    common.plot_category_component_weights(
        S.factors_adata(),
        component,
        spec["group_column"],
        ax,
        xlabel=association.split(" (")[0],  # the levels show the grouping
        anova_df=table[table["Association"] == association],
        patient_column=S.SPEC.patient_column,
        condition_column=S.SPEC.condition_column,
        show_counts=False,
        **{k: spec[k] for k in REGROUPING_KEYS if k in spec},
    )
    ax.set_ylabel(f"CP component {component} loading")
    ax.tick_params(axis="x", labelsize=style.FONT_SIZE_DENSE)


def makeFigure():
    return common.standalone_fixed(__name__)
