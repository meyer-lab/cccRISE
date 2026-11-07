"""Full-data CP fit R2X with and without the masked-imputation ridge (GPU),
and the canonical CPU core fit's at the selected rank."""

from .. import common, style
from . import _common as S

AXES = ("square",)


def plot(axs):
    df, core, l2_reg = S.load()
    ax = axs[0]
    levels = common.per_rank(df)
    for column, color, label in (
        ("NoRidgeFitR2X", style.COLORS["fit"], "No ridge (GPU)"),
        (
            "RidgeFitR2X",
            style.COLORS["highlight"],
            rf"Ridge, $\lambda$ = {l2_reg:g} (GPU)",
        ),
    ):
        ax.plot(levels["Rank"], levels[column], marker="o", color=color, label=label)
    ax.scatter(
        core["Rank"],
        core["CoreFitR2X"],
        marker="x",
        color=style.COLORS["ink"],
        label="Core fit (CPU)",
        **S.CORE_MARKER,
    )
    common.mark_rank(ax, core["Rank"])
    ax.set_xlabel("CP rank")
    ax.set_ylabel("Fit R²X")
    common.legend(ax, "below")


def makeFigure():
    return common.standalone_fixed(__name__)
