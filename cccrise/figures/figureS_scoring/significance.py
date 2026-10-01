"""Permutation significance under each score: mean fraction of significant
entries per LR pair, and top-K overlap of the most significant LR pairs
with the default's."""

import pandas as pd

from .. import common, style
from . import _shared as S


def significance_data() -> tuple[pd.Series, pd.DataFrame]:
    fraction, ranked = {}, {}
    for method in S.ALL_SCORES:
        table = S.load_method_output(S.SPEC.significance_lr_filename(), method)
        fraction[method] = table["FractionSignificant_p05"].mean()
        # The largest positive z-scores are the most significant.
        ranked[method] = table.sort_values("MeanZScore", ascending=False)["LRPair"]
    overlaps = pd.DataFrame(
        [
            {
                "Method": method,
                "K": k,
                "Jaccard": S.jaccard(
                    set(ranked[S.DEFAULT_SCORE].head(k)), set(ranked[method].head(k))
                ),
            }
            for method in S.ALT_SCORES
            for k in S.LR_TOP_KS
        ]
    )
    return pd.Series(fraction), overlaps


def plot(fig):
    fraction, overlaps = significance_data()
    axs = fig.subplots(1, 2)
    axs[0].bar(
        range(len(fraction)),
        fraction,
        color=[style.COMMUNICATION_SCORE_COLORS[m] for m in fraction.index],
    )
    axs[0].set_xticks(
        range(len(fraction)),
        [S.SCORE_LABELS[m] for m in fraction.index],
        rotation=30,
        ha="right",
    )
    common.reference_line(axs[0], common.SIG_ALPHA)
    axs[0].set_ylabel("Mean fraction significant\nper LR pair (p < 0.05)")
    axs[0].set_title("Permutation significance")
    S.draw_top_k_overlap(axs[1], overlaps, "K (top LR pairs by z-score)")
    axs[1].set_title("Most significant LR pairs")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"], legend=S.method_legend)
