"""Figure 3F's test for C1 and C5: each top LR pair's CD-inflamed minus
healthy mean depth-normalized context score against `run_ibd_programs`'
biopsy-label permutations, in units of the null's SD (violins: null;
diamonds: observed, filled where the BH q across the component's top pairs
is below `common.SIG_ALPHA`)."""

import numpy as np
from scipy.stats import false_discovery_control

from ... import common, style
from .. import _common as S

AXES = ("rect", "rect")


def plot(axs):
    table = S.programs("program_permutation")
    summary = table.drop_duplicates(["Component", "LR pair"]).sort_values(
        ["Component", "Rank"]
    )
    summary = summary.assign(
        q=summary.groupby("Component")["P"].transform(
            lambda p: false_discovery_control(p.to_numpy())
        )
    )
    common.source_data(
        "program_top_pair_permutation",
        summary[["Component", "Rank", "LR pair", "Observed", "P", "q"]],
        "Each C1/C5 top LR pair's CD inflamed minus healthy mean depth-normalized "
        "context score, one-sided permutation p over biopsy-label permutations, and "
        "BH q across the component's top pairs.",
    )
    for ax, k in zip(axs, S.PROGRAM_COLORS, strict=True):
        rows = summary[summary["Component"] == k]
        null = table[table["Component"] == k].pivot(
            index="Permutation", columns="LR pair", values="Null difference"
        )
        null = null[rows["LR pair"]].to_numpy()
        sd, centre = null.std(0), null.mean(0)
        y = np.arange(len(rows))
        bodies = ax.violinplot(
            (null - centre) / sd, positions=y, vert=False, showextrema=False, widths=0.8
        )
        for body in bodies["bodies"]:
            body.set(facecolor=style.COLORS["muted"], alpha=0.6, edgecolor="none")
        significant = rows["q"].to_numpy() < common.SIG_ALPHA
        ax.scatter(
            (rows["Observed"].to_numpy() - centre) / sd,
            y,
            marker="D",
            s=style.POINT_SIZE,
            facecolor=np.where(significant, S.PROGRAM_COLORS[k], "white"),
            edgecolor=S.PROGRAM_COLORS[k],
            zorder=3,
        )
        ax.set_yticks(
            y,
            [p.replace("^", "–") for p in rows["LR pair"]],
            fontsize=style.FONT_SIZE_DENSE,
        )
        ax.invert_yaxis()
        ax.set_xscale("symlog", linthresh=3)
        ax.set_xlabel("Inflamed − healthy (null SDs)")
        ax.set_title(f"C{k} top pairs", fontsize=style.FONT_SIZE_LABEL)


def makeFigure():
    return common.standalone_fixed(__name__)
