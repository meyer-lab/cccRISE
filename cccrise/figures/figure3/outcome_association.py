"""Condition loadings of the CP component associated with the 6-month
outcome, per outcome level, with a patient's paired samples joined."""

import pandas as pd

from .. import common, style
from . import _common as S

REGROUPING_KEYS = ("levels", "mapping", "control_label", "other_label")
AXES = ("square",)
ANOVA_COLUMNS = [
    "Association",
    "Component",
    "F",
    "P",
    "PAdj",
    "EtaSquared",
    "NGroups",
    "GroupSizes",
]


def plot(axs):
    plot_outcome_weights(axs[0], S.OUTCOME, S.COMPONENT)
    # Angled, as the four levels crowd the square axes
    for label in axs[0].get_xticklabels():
        label.set(rotation=40, ha="right", rotation_mode="anchor")
    _source_data(S.OUTCOME, S.COMPONENT)


def plot_outcome_weights(ax, association: str, component: int) -> None:
    """CP `component`'s condition loadings per level of `association`, with
    its ANOVA statistics from `S.association_table`. Sample counts are left
    to the caption."""
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
    ax.set_ylabel(f"CPD component {component} loading")
    ax.tick_params(axis="x", labelsize=style.FONT_SIZE_DENSE)


def outcome_points(association: str, component: int) -> pd.DataFrame:
    """The samples `plot_outcome_weights` draws: each one's loading and
    level, with its patient."""
    spec = S.association(association)
    points = common.group_loadings(
        S.factors_adata(),
        component,
        spec["group_column"],
        S.SPEC.condition_column,
        **{k: spec[k] for k in REGROUPING_KEYS if k in spec},
    )
    patients = S.condition_obs()[S.SPEC.patient_column]
    return points.assign(Patient=patients.loc[points["Condition"]].to_numpy())


def _source_data(association: str, component: int) -> None:
    """The plotted loadings, with each sample's patient, and their ANOVA."""
    common.source_data(
        "cp_loadings_by_outcome",
        outcome_points(association, component),
        f"Each sample's read-depth-corrected loading on CPD component {component} "
        f"and its level of '{association}', with its patient (a patient's "
        "samples are joined in the plot).",
    )
    table = S.association_table()
    row = table[(table["Association"] == association)]
    common.source_data(
        "cp_outcome_anova",
        row[row["Component"] == component][ANOVA_COLUMNS],
        "One-way ANOVA of the loadings over the outcome levels: F, p, p "
        "Benjamini-Hochberg adjusted across every component and association "
        "tested (PAdj), and eta-squared.",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
