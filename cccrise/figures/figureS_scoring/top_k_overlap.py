"""Top-K overlap of each alternate score's matched components with the
default's: LR pairs by |D| loading, and sender/receiver cell types by mean
per-cell loading (needs the per-cell projection)."""

import pandas as pd

from .. import style
from . import _shared as S


def lr_overlaps() -> pd.DataFrame:
    ref, lr_pairs = S.cp_factors(S.DEFAULT_SCORE)
    rows = []
    for method in S.ALT_SCORES:
        alt, _ = S.cp_factors(method)
        for i, j in S.matched_pairs(method):
            rows += [
                {"Method": method, "Component": i + 1} | overlap
                for overlap in S.top_k_jaccard(
                    pd.Series(ref[3][:, i], lr_pairs),
                    pd.Series(alt[3][:, j], lr_pairs),
                    S.LR_TOP_KS,
                )
            ]
    return pd.DataFrame(rows)


def celltype_overlaps() -> pd.DataFrame:
    """Mean of the sender and receiver cell-type overlaps."""
    ref = S.celltype_loadings(S.DEFAULT_SCORE)
    rows = []
    for method in S.ALT_SCORES:
        alt = S.celltype_loadings(method)
        for i, j in S.matched_pairs(method):
            sender, receiver = (
                S.top_k_jaccard(r.iloc[:, i], a.iloc[:, j], S.CELLTYPE_KS)
                for r, a in zip(ref, alt, strict=True)
            )
            rows += [
                {
                    "Method": method,
                    "Component": i + 1,
                    "K": s["K"],
                    "Jaccard": (s["Jaccard"] + r["Jaccard"]) / 2,
                }
                for s, r in zip(sender, receiver, strict=True)
            ]
    return pd.DataFrame(rows)


def plot(fig):
    axs = fig.subplots(1, 2)
    S.draw_top_k_overlap(axs[0], lr_overlaps(), "K (top LR pairs)")
    S.draw_top_k_overlap(axs[1], celltype_overlaps(), "K (top cell types)")
    axs[0].set_title("LR pairs")
    axs[1].set_title("Sender/receiver cell types")


def makeFigure():
    return S.standalone(plot, style.PANEL_SIZE["half"])
