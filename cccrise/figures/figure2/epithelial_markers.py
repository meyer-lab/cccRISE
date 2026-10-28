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
# marker shown, leaving out significant differences of a handful of cells
MIN_DETECTED_DIFFERENCE = 0.2


def plot(axs):
    markers = stratifying_markers() | {"Top LR pairs": S.top_pair_genes()}
    S.plot_markers(
        axs[0],
        # {"Pan-epithelial": S.CELLTYPE_MARKERS["Epithelial cells"]} |
        markers,
        show_legend=True,
    )
    common.source_data(
        "marker_expression_by_group",
        _dotplot_data(markers),
        "Per marker gene (under its section) and group: the fraction of the "
        "group's cells detecting it (dot size) and its mean normalized "
        "expression, also relative to the gene's largest group mean (color). "
        "Each cell's group is in this figure's sender vs. receiver scatter "
        "panel's source data; expression is from the published dataset.",
    )


def _dotplot_data(markers: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for section, genes in markers.items():
        detected, mean = S.group_expression(S.present(genes))
        relative = mean.div(mean.max(axis=1), axis=0)
        for stat, name in (
            (detected, "Fraction detected"),
            (mean, "Mean expression"),
            (relative, "Mean expression / max group mean"),
        ):
            rows.append(
                stat.rename_axis(index="Gene", columns="Group")
                .stack()
                .rename(name)
                .reset_index()
                .assign(Section=section)
            )
    keys = ["Section", "Gene", "Group"]
    out = pd.concat(rows).groupby(keys, sort=False).first()
    return out.reset_index()


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
    q = pd.Series(false_discovery_control(p), index=tested)
    kept = (q < common.SIG_ALPHA) & (
        np.abs(detected["sender"] - detected["receiver"]) >= MIN_DETECTED_DIFFERENCE
    )
    selected = {
        name: [g for g in genes if kept[g]] for name, genes in candidates.items()
    }
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
    return {name: genes for name, genes in selected.items() if genes}


def makeFigure():
    return common.standalone_fixed(__name__)
