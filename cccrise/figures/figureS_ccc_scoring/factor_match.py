"""How closely each alternate score's CP components match the default's:
the pairing's match score (condition + LR modes) and the paired components'
sender/receiver |cosine|, an independent check of the pairing."""

import numpy as np
import pandas as pd

from .. import common, style
from . import _common as S


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def similarity_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(component x alternate method) match scores and mean B/C |cosine|."""
    ref, _ = S.cp_factors(S.DEFAULT_SCORE)
    match, cosine = {}, {}
    for method in S.ALT_SCORES:
        alt, _ = S.cp_factors(method)
        scores = S.match_components(method).set_index("Component")["MatchScore"]
        match[method] = scores
        cosine[method] = pd.Series(
            [
                np.mean([_cosine(ref[m][:, i], alt[m][:, j]) for m in (1, 2)])
                for i, j in S.matched_pairs(method)
            ],
            index=scores.index,
        )
    return pd.DataFrame(match), pd.DataFrame(cosine)


def plot(fig):
    match, cosine = similarity_tables()
    axs = fig.subplots(1, 2)
    S.grouped_bar(axs[0], match, "Match score")
    S.grouped_bar(axs[1], cosine, "Sender/receiver |cosine|")
    for ax in axs:
        ax.set_ylim(0, 1.05)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"], legend=S.method_legend)
