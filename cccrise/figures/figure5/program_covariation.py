"""Figure 3D's test for the inflammatory programs C1 and C5 on one axis:
the Spearman correlation across biopsies of every pair of each component's
top LR pairs' context scores, beside the same for random,
expression-matched and other status-associated pairs
(`covariation.covariation_baselines`; `run_ibd_programs`)."""

from .. import common, style
from ..figure3.coordinated_changes import BASELINES
from . import _common as S

AXES = ("half",)
SHORT = {"Expression-matched": "Expr.-matched", "Other associated": "Other assoc."}


def plot(axs):
    (ax,) = axs
    table = S.programs("program_covariation").dropna(subset=["Rank correlation"])
    common.source_data(
        "program_pair_covariation",
        table,
        "Pairwise Spearman correlation across biopsies of each component's top "
        "LR pairs' context scores ('Top pairs'), and of every baseline draw's "
        "pairs (random, expression-matched, other status-associated; 2,000 draws).",
    )
    ticks, labels = [], []
    for block, k in enumerate(S.PROGRAM_COLORS):
        rows = table[table["Component"] == k]
        sets = {"Top pairs": (f"C{k} top 10", S.PROGRAM_COLORS[k])} | {
            name: (SHORT.get(label, label), color)
            for name, (label, color) in BASELINES.items()
        }
        x = [block * (len(sets) + 1) + i for i in range(len(sets))]
        boxes = ax.boxplot(
            [rows.loc[rows["Set"] == name, "Rank correlation"] for name in sets],
            positions=x,
            widths=0.65,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": style.COLORS["ink"]},
        )
        for box, (_, color) in zip(boxes["boxes"], sets.values(), strict=True):
            box.set(facecolor=color, alpha=0.7)
        ticks += x
        labels += [label for label, _ in sets.values()]
    common.reference_line(ax, 0)
    ax.set_xticks(
        ticks,
        labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.set_ylabel("Pairwise rank correlation\nacross biopsies")


def makeFigure():
    return common.standalone_fixed(__name__)
