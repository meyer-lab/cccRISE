"""Condition loadings of the component most associated with the shown
outcome, per outcome level, with a patient's paired samples joined."""

from .. import common, style
from . import _common as S

#: Associations (by `ASSOCIATIONS` name) shown, each with its most strongly
#: associated component.
SHOWN = ["6-Month Outcome (4-Way)"]
REGROUPING_KEYS = ("levels", "mapping", "control_label", "other_label")
# More levels than this get rotated tick labels to fit a standard-width axes.
MAX_UPRIGHT_LEVELS = 2


AXES = ("square",) * len(SHOWN)


def plot(axs):
    table = S.association_table()
    for ax, name in zip(axs, SHOWN, strict=True):
        spec = S.association(name)
        rows = table[table["Association"] == name]
        component = int(rows.loc[rows["PAdj"].idxmin(), "Component"])
        common.plot_category_component_weights(
            S.factors_adata(),
            component,
            spec["group_column"],
            ax,
            xlabel=name.split(" (")[0],  # the levels show the grouping
            anova_df=rows,
            patient_column=S.SPEC.patient_column,
            condition_column=S.SPEC.condition_column,
            **{k: spec[k] for k in REGROUPING_KEYS if k in spec},
        )
        labels = [
            t.get_text().replace("/\n", "/").replace("\n", " ")
            for t in ax.get_xticklabels()
        ]
        if len(labels) > MAX_UPRIGHT_LEVELS:
            ax.set_xticks(
                ax.get_xticks(), labels, rotation=45, ha="right", rotation_mode="anchor"
            )
        ax.tick_params(axis="x", labelsize=style.FONT_SIZE_DENSE)


def makeFigure():
    return common.standalone_fixed(__name__)
