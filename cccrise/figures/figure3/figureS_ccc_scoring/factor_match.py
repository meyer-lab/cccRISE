"""Match of each alternate communication score's CPD components to the default
score's: their mean |cosine| over the condition and LR pair modes, and the
absolute cosine of the paired components' sender and receiver loadings."""

import numpy as np
import pandas as pd

from ... import common, style
from . import _common as S


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def similarity() -> pd.DataFrame:
    """One row per (alternate method, default component): its matched
    component, the match score, and the paired components' sender (B) and
    receiver (C) |cosine| and their mean."""
    ref, _ = S.cp_factors(S.DEFAULT_SCORE)
    frames = []
    for method in S.ALT_SCORES:
        alt, _ = S.cp_factors(method)
        frame = S.match_components(method)[
            ["Component", "MatchedComponent", "MatchScore"]
        ].assign(Method=S.SCORE_LABELS[method])
        for mode, name in ((1, "SenderCosine"), (2, "ReceiverCosine")):
            frame[name] = [
                _cosine(ref[mode][:, i], alt[mode][:, j])
                for i, j in S.matched_pairs(method)
            ]
        frame["MeanCosine"] = frame[["SenderCosine", "ReceiverCosine"]].mean(axis=1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def similarity_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(component x alternate method) match scores and mean B/C |cosine|,
    saved as source data."""
    table = similarity()
    common.source_data(
        "factor_match_scores",
        table[["Method", "Component", "MatchedComponent", "MatchScore"]],
        "One row per (alternate score, default-score CPD component): the "
        "alternate score's paired component and their match score (mean "
        "|cosine| over the condition and LR pair modes).",
        part=0,
    )
    common.source_data(
        "factor_match_cosines",
        table.drop(columns="MatchScore"),
        "One row per (alternate score, default-score CPD component): the |cosine| "
        "between the paired components' sender (B) and receiver (C) eigen-state "
        "loadings, and their mean (the bars).",
        part=1,
    )
    methods = {S.SCORE_LABELS[m]: m for m in S.ALT_SCORES}
    match, cosine = (
        table.pivot(index="Component", columns="Method", values=v).rename(
            columns=methods
        )[S.ALT_SCORES]
        for v in ("MatchScore", "MeanCosine")
    )
    return match, cosine


def plot(fig):
    match, cosine = similarity_tables()
    axs = fig.subplots(1, 2)
    S.grouped_bar(axs[0], match, "Condition/LR |cosine|\nwith product score")
    S.grouped_bar(axs[1], cosine, "Sender/receiver |cosine|")
    for ax in axs:
        ax.set_ylim(0, 1.05)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"], legend=S.method_legend)
