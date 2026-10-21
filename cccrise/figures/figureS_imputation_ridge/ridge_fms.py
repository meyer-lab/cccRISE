"""Component-matched FMS between the full-data CP fits with and without the
masked-imputation ridge (GPU): each matched component's congruence product and
their mean; and, at the selected rank, the canonical CPU core fit's FMS to
each."""

from .. import common, style
from . import _common as S

AXES = ("square",)


def plot(axs):
    df, core, _ = S.load()
    ax = axs[0]
    ax.scatter(
        df["Rank"],
        df["RidgeComponentFMS"],
        s=style.POINT_SIZE_BACKGROUND,
        color=style.COLORS["muted"],
        label="Component",
    )
    levels = common.per_rank(df)
    ax.plot(
        levels["Rank"],
        levels["RidgeFMS"],
        color=style.COLORS["ink"],
        label="Ridge vs. no ridge",
    )
    for column, marker, color, label in (
        ("CoreFMS", "D", style.COLORS["fit"], "Core vs. no ridge"),
        ("CoreRidgeFMS", "s", style.COLORS["highlight"], "Core vs. ridge"),
    ):
        ax.scatter(
            core["Rank"],
            core[column],
            marker=marker,
            edgecolors=color,
            label=label,
            **S.CORE_MARKER,
        )
    common.mark_rank(ax, core["Rank"])
    ax.set_xlabel("CP rank")
    ax.set_ylabel("FMS")
    ax.set_ylim(-0.03, 1.05)
    common.legend(ax, "below")


def makeFigure():
    return common.standalone_fixed(__name__)
