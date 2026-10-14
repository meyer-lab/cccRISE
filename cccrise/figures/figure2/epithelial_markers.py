"""Label-free identity of the granularity component's sender and receiver
groups: canonical pan-epithelial markers (`S.CELLTYPE_MARKERS`), the canonical
epithelial subtype markers of the same atlas (`S.EPITHELIAL_SUBTYPE_MARKERS`)
that stratify sender from receiver (`stratifying_markers`), and the genes of
the component's top LR pairs, per group."""

import pandas as pd
from scipy.stats import false_discovery_control

from .. import common, style
from . import _common as S

AXES = ("strip",)


def plot(axs):
    S.plot_markers(
        axs[0],
        {"Pan-epithelial": S.CELLTYPE_MARKERS["Epithelial cells"]}
        | stratifying_markers()
        | {"Top LR pairs": S.top_pair_genes()},
        show_legend=True,
        fontsize=style.FONT_SIZE_TICK,  # as the group labels
    )


def stratifying_markers() -> dict[str, list[str]]:
    """{subtype: its markers whose sender vs. receiver Wilcoxon p, BH-adjusted
    over the subtype markers in the fit's gene set alone (the pre-specified
    hypotheses, not the genome), is below `SIG_ALPHA`}, subtypes without one
    left out."""
    candidates = {
        name: S.present(genes) for name, genes in S.EPITHELIAL_SUBTYPE_MARKERS.items()
    }
    tested = S.present(g for genes in candidates.values() for g in genes)
    p = S.rank_genes(("sender",), "receiver")["sender"].loc[tested, "p"]
    q = pd.Series(false_discovery_control(p), index=tested)
    selected = {
        name: [g for g in genes if q[g] < common.SIG_ALPHA]
        for name, genes in candidates.items()
    }
    return {name: genes for name, genes in selected.items() if genes}


def makeFigure():
    return common.standalone_fixed(__name__)
