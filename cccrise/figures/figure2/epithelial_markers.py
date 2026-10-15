"""Label-free identity of the granularity component's sender and receiver
groups: canonical pan-epithelial markers (`S.CELLTYPE_MARKERS`), the
epithelial subtype markers of two lung atlases (`S.EPITHELIAL_SUBTYPE_MARKERS`)
that stratify sender from receiver (`stratifying_markers`), and the genes of
the component's top LR pairs, per group."""

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from .. import common
from . import _common as S

AXES = ("strip",)
# Least sender - receiver difference in the fraction of cells detecting a
# marker shown: the tie-corrected test calls differences of a handful of cells
MIN_DETECTED_DIFFERENCE = 0.2


def plot(axs):
    S.plot_markers(
        axs[0],
        {"Pan-epithelial": S.CELLTYPE_MARKERS["Epithelial cells"]}
        | stratifying_markers()
        | {"Top LR pairs": S.top_pair_genes()},
        show_legend=True,
    )


def stratifying_markers() -> dict[str, list[str]]:
    """{subtype: its markers whose sender vs. receiver Wilcoxon p, BH-adjusted
    over every subtype marker in the fit's gene set alone (the pre-specified
    hypotheses, not the genome), is below `SIG_ALPHA` and whose detected
    fractions differ by `MIN_DETECTED_DIFFERENCE`}, subtypes without one left
    out."""
    candidates = {
        name: S.present(genes) for name, genes in S.EPITHELIAL_SUBTYPE_MARKERS.items()
    }
    tested = S.present(g for genes in candidates.values() for g in genes)
    p = S.rank_genes(("sender",), "receiver")["sender"].loc[tested, "p"]
    detected, _ = S.group_expression(tested)
    kept = (pd.Series(false_discovery_control(p), index=tested) < common.SIG_ALPHA) & (
        np.abs(detected["sender"] - detected["receiver"]) >= MIN_DETECTED_DIFFERENCE
    )
    selected = {
        name: [g for g in genes if kept[g]] for name, genes in candidates.items()
    }
    return {name: genes for name, genes in selected.items() if genes}


def makeFigure():
    return common.standalone_fixed(__name__)
