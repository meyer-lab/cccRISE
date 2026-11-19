"""Each CPD component's 20 most heavily weighted LR pairs: log2 ratio of the
mean depth-normalized score in CLAD/died to that in stable/recovered samples
against the one-sided patient-permutation p.

Every CPD component's `N_SHOWN` most heavily weighted LR pairs, each in its
own component's context (`lr_permutation`): log2 ratio of the group means
of the depth-normalized scores against the pair's one-sided permutation p.
The outcome-associated component's top `N_TOP` pairs are highlighted and
named where p < 0.05, and its next `N_SHOWN - N_TOP` set off from the other
components' pairs."""

import numpy as np
import pandas as pd
from adjustText import adjust_text

from .. import common, style
from ._common import COMPONENT
from .coordinated_changes import N_TOP
from .lr_permutation import (
    CASE_DISPLAY,
    CONTROL_DISPLAY,
    LR_NAMES,
    N_SHOWN,
    permutation_analysis,
    save_permutation_tests,
)

AXES = ("square",)
# Labels are right-aligned in a column at this log2 ratio, in the empty upper
# left, and spread vertically by adjustText
LABEL_X = -0.3
LABEL_GAP = 0.1  # between a label and the start of its leader line


def plot(axs):
    (ax,) = axs
    result = permutation_analysis()
    points = {
        k: (
            mean_log2_ratio(r)[r["order"][:N_SHOWN]],
            -np.log10(r["p"][r["order"][:N_SHOWN]]),
        )
        for k, r in result.items()
    }
    save_permutation_tests(result)
    common.source_data(
        "top_pair_log2_ratios",
        pd.concat(
            pd.DataFrame(
                {
                    "Component": k,
                    "LR pair": LR_NAMES[r["order"][:N_SHOWN]],
                    "log2 mean ratio": points[k][0],
                }
            )
            for k, r in result.items()
        ),
        f"The log2 ratio of the {CASE_DISPLAY} to the {CONTROL_DISPLAY} mean "
        "score plotted for every CPD component's top pairs, the means offset by "
        "1e-3 x the component's median positive mean.",
    )
    others = [v for k, v in points.items() if k != COMPONENT]
    x, y = points[COMPONENT]
    tiers = [  # x, y, size, color, label
        (
            *(np.concatenate(v) for v in zip(*others, strict=True)),
            style.POINT_SIZE,
            style.COLORS["muted"],
            "Other cmps.",
        ),
        (
            x[N_TOP:],
            y[N_TOP:],
            style.POINT_SIZE,
            style.COLORS["ink"],
            f"Cmp. {COMPONENT}, {N_TOP + 1}–{N_SHOWN}",
        ),
        (
            x[:N_TOP],
            y[:N_TOP],
            style.POINT_SIZE_FOREGROUND,
            style.COLORS["emphasis"],
            f"Cmp. {COMPONENT}, 1–{N_TOP}",
        ),
    ]
    for tx, ty, size, color, label in tiers:
        ax.scatter(tx, ty, s=size, color=color, linewidths=0, label=label)
    common.reference_line(ax, -np.log10(common.SIG_ALPHA))
    common.reference_line(ax, 0, horizontal=False, linestyle=":")
    label_significant(
        ax,
        x[:N_TOP],
        y[:N_TOP],
        LR_NAMES[result[COMPONENT]["order"][:N_TOP]],
        *(np.concatenate(v) for v in zip(*points.values(), strict=True)),
    )
    ax.set(
        xlabel=f"log$_2$(mean score ratio)\n{CASE_DISPLAY} ÷ {CONTROL_DISPLAY}",
        ylabel="−log$_{10}$(permutation p)",
    )
    common.legend(
        ax,
        "above",
        ncols=2,
        title=f"Top {N_SHOWN} LR pairs by loading",
        fontsize=style.FONT_SIZE_DENSE,
        title_fontsize=style.FONT_SIZE_DENSE,
    )


def mean_log2_ratio(component: dict) -> np.ndarray:
    """log2 ratio of the group means, offset by a thousandth of the median
    positive one so pairs absent from a group stay finite."""
    case, control = component["means"]
    both = np.concatenate([case, control])
    offset = 1e-3 * np.median(both[both > 0])
    return np.log2((case + offset) / (control + offset))


def label_significant(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    names: np.ndarray,
    all_x: np.ndarray,
    all_y: np.ndarray,
) -> None:
    """Name the points above the p = 0.05 line in a column left of them, in
    their vertical order, evenly spaced and kept clear of every plotted point
    (`all_x`, `all_y`) by adjustText, each joined to its point."""
    shown = np.flatnonzero(y > -np.log10(common.SIG_ALPHA))
    shown = shown[np.argsort(-y[shown])]
    heights = np.linspace(y[shown[0]], -np.log10(common.SIG_ALPHA), len(shown))
    texts = [
        ax.text(
            LABEL_X,
            h,
            names[i],
            ha="right",
            va="center",
            fontsize=common.FACTOR_LABEL_FONT_SIZE,
        )
        for h, i in zip(heights, shown, strict=True)
    ]
    adjust_text(
        texts,
        x=all_x,
        y=all_y,
        ax=ax,
        only_move={"text": "y", "static": "y", "explode": "y", "pull": "y"},
        time_lim=10,
    )
    for text, i in zip(texts, shown, strict=True):
        ax.plot(
            [LABEL_X + LABEL_GAP, x[i]],
            [text.get_position()[1], y[i]],
            color=style.COLORS["reference"],
            lw=style.LINE_WIDTH_THIN / 2,
        )


def makeFigure():
    return common.standalone_fixed(__name__)
