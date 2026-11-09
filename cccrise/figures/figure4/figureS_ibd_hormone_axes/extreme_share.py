"""Per biopsy (at least `_common.MIN_EE` EE cells), the share of its EE cells
in each class of Figure 4's marker dot plot (`_common.hormone_extremes`), by
inflammation status (Kruskal-Wallis q, BH corrected across classes, drawn and in
the source data; pairwise Mann-Whitney q in the source data): inflammation removes
villus-like cells, not crypt-like ones."""

import pandas as pd
from scipy.stats import mannwhitneyu

from ....anova import adjust_pvalues
from ... import common
from .. import _common as S

AXES = ("rect",)
#: Pairwise tests in the source data: (label, status a, status b)
PAIRS = [("U–I", "CD_uninflamed", "CD_inflamed"), ("H–I", "Healthy", "CD_inflamed")]


def plot(axs):
    (ax,) = axs
    shares = S.class_shares(S.hormone_extremes)
    groups = {
        name: {g: shares.loc[shares["Status"] == g, name] for g in S.GROUP_LABELS}
        for name in S.EXTREME_NAMES
    }
    common.source_data(
        "hormone_extreme_share_by_biopsy",
        shares.rename_axis("Biopsy").reset_index(),
        f"Share of each biopsy's EE cells (biopsies with at least {S.MIN_EE}) in "
        "each class, with the biopsy's inflammation status; Kruskal-Wallis q "
        "across statuses (Benjamini-Hochberg corrected across classes) above "
        "each class.",
    )
    pairwise = pd.DataFrame(
        [
            {
                "Class": name,
                "Test": label,
                "p": mannwhitneyu(values[a], values[b]).pvalue,
            }
            for name, values in groups.items()
            for label, a, b in PAIRS
        ]
    )
    common.source_data(
        "hormone_extreme_share_pairwise",
        pairwise.assign(q=adjust_pvalues(pairwise["p"])),
        "Mann-Whitney p of each class's per-biopsy share between CD uninflamed (U) "
        "or healthy (H) and CD inflamed (I) biopsies, and its Benjamini-Hochberg "
        "q across all classes and pairs.",
    )
    q = S.status_boxes(
        ax,
        {
            S.EXTREME_LABELS[n].replace(" (", "\n("): list(values.values())
            for n, values in groups.items()
        },
    )
    common.source_data(
        "hormone_extreme_share_kruskal",
        pd.DataFrame({"Class": list(groups), "q": q}),
        "Kruskal-Wallis q of each class's per-biopsy share across healthy, CD "
        "uninflamed, and CD inflamed biopsies (Benjamini-Hochberg corrected across "
        "classes), as drawn above each class.",
    )
    ax.set_ylabel("Share of biopsy's\nEE cells")
    S.status_legend(
        ax,
        loc="upper left",
        labelspacing=0.2,
        borderaxespad=0.2,
        borderpad=0.2,
        handlelength=1,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
