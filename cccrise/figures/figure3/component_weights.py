"""Condition loadings of the component most associated with each shown
outcome, per outcome level, with a patient's paired samples joined."""

from .. import common, style
from . import _common as S

#: Associations (by `ASSOCIATIONS` name) shown, each with its most strongly
#: associated component.
SHOWN = ["6-Month Outcome (4-Way)", "Acute Rejection (Yes vs. No)"]
REGROUPING_KEYS = ("levels", "mapping", "control_label", "other_label")


def plot(fig):
    table = S.association_table()
    # Axis widths follow each outcome's number of levels.
    n_levels = [
        len(table.loc[table["Association"] == name, "GroupSizes"].iloc[0])
        for name in SHOWN
    ]
    axs = fig.subplots(1, len(SHOWN), width_ratios=n_levels)
    for ax, name in zip(axs, SHOWN, strict=True):
        spec = S.association(name)
        rows = table[table["Association"] == name]
        component = int(rows.loc[rows["PAdj"].idxmin(), "Component"])
        common.plot_category_component_weights(
            S.factors_adata(),
            component,
            spec["group_column"],
            ax,
            title=name.split(" (")[0],  # the levels show the grouping
            anova_df=rows,
            patient_column=S.SPEC.patient_column,
            condition_column=S.SPEC.condition_column,
            **{k: spec[k] for k in REGROUPING_KEYS if k in spec},
        )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
