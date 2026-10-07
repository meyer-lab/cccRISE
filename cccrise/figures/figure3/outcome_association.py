"""Condition loadings of the CP component most associated with the 6-month
outcome, per outcome level, with a patient's paired samples joined."""

from .. import common, style
from . import _common as S

REGROUPING_KEYS = ("levels", "mapping", "control_label", "other_label")
# More levels than this get rotated tick labels to fit a standard-width axes.
MAX_UPRIGHT_LEVELS = 2
AXES = ("square",)


def plot(axs):
    # TODO: plots read-depth-corrected loadings, but `S.association_table`
    # tests the uncorrected cp_A, so the boxes are not what was tested.
    plot_outcome_weights(axs[0], S.factors_adata(), S.association_table(), "CP")


def plot_outcome_weights(
    ax, adata, table, kind: str, read_depth_correct: bool = True
) -> None:
    """The `kind` ("CP" or "RISE") component most associated with
    `S.OUTCOME` in `table`: its condition loadings (`adata.uns["A"]`) per
    outcome level, with the ANOVA statistics."""
    spec = S.association(S.OUTCOME)
    rows = table[table["Association"] == S.OUTCOME]
    component = int(rows.loc[rows["PAdj"].idxmin(), "Component"])
    common.plot_category_component_weights(
        adata,
        component,
        spec["group_column"],
        ax,
        xlabel=S.OUTCOME.split(" (")[0],  # the levels show the grouping
        anova_df=rows,
        patient_column=S.SPEC.patient_column,
        condition_column=S.SPEC.condition_column,
        read_depth_correct=read_depth_correct,
        **{k: spec[k] for k in REGROUPING_KEYS if k in spec},
    )
    ax.set_ylabel(f"{kind} component {component} loading")
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
