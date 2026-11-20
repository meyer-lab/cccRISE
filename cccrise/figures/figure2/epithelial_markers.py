"""Label-free identity of the granularity component's sender and receiver
groups: canonical pan-epithelial markers (`S.CELLTYPE_MARKERS`), the
epithelial subtype markers of two lung atlases (`S.EPITHELIAL_SUBTYPE_MARKERS`)
that stratify sender from receiver (`stratifying_markers`), and the genes of
the component's top LR pairs, per group."""

import functools

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from .. import common, style
from . import _common as S

AXES = ("strip",)
# Least sender - receiver difference in the fraction of cells detecting a
# marker shown, leaving out significant differences of a handful of cells
MIN_DETECTED_DIFFERENCE = 0.2


def plot(axs):
    _save_tests(*marker_tests())
    S.plot_markers(
        axs[0],
        # {"Pan-epithelial": S.CELLTYPE_MARKERS["Epithelial cells"]} |
        stratifying_markers() | {"Top LR pairs": S.top_pair_genes()},
        show_legend=True,
        fontsize=style.FONT_SIZE_TICK,
    )


def stratifying_markers() -> dict[str, list[str]]:
    """{subtype: its markers whose sender vs. receiver Wilcoxon p, BH-adjusted
    over every subtype marker in the fit's gene set alone (the pre-specified
    hypotheses, not the genome), is below `SIG_ALPHA` and whose detected
    fractions differ by `MIN_DETECTED_DIFFERENCE`}, subtypes without one left
    out."""
    candidates, *_, kept = marker_tests()
    selected = {
        name: [g for g in genes if kept[g]] for name, genes in candidates.items()
    }
    return {name: genes for name, genes in selected.items() if genes}


@functools.cache
def marker_tests():
    """(candidate markers per subtype, and over every tested marker: Wilcoxon
    p, BH q, detected fractions per group, and whether shown) behind
    `stratifying_markers`."""
    candidates = {
        name: S.present(genes) for name, genes in S.EPITHELIAL_SUBTYPE_MARKERS.items()
    }
    tested = S.present(g for genes in candidates.values() for g in genes)
    p = S.rank_genes(("sender",), "receiver")["sender"].loc[tested, "p"]
    detected, _ = S.group_expression(tested)
    q = pd.Series(false_discovery_control(p), index=tested)
    kept = (q < common.SIG_ALPHA) & (
        np.abs(detected["sender"] - detected["receiver"]) >= MIN_DETECTED_DIFFERENCE
    )
    return candidates, p, q, detected, kept


def _save_tests(candidates, p, q, detected, kept) -> None:
    common.source_data(
        "de_sender_vs_receiver_subtype_markers",
        pd.DataFrame(
            {
                "Subtype": name,
                "Gene": g,
                "Wilcoxon p": p[g],
                "BH q": q[g],
                "Fraction detected (sender)": detected.loc[g, "sender"],
                "Fraction detected (receiver)": detected.loc[g, "receiver"],
                "Shown": kept[g],
            }
            for name, genes in candidates.items()
            for g in genes
        ),
        "Every epithelial subtype marker tested, sender against receiver cells: "
        "two-sided Wilcoxon rank-sum p, Benjamini-Hochberg adjusted over all "
        f"markers tested. Shown if q < {common.SIG_ALPHA} and the detected "
        f"fractions differ by at least {MIN_DETECTED_DIFFERENCE}.",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
