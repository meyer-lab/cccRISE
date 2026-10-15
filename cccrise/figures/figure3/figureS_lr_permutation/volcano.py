"""Every CP component's `N_SHOWN` most heavily weighted LR pairs, each in its
own component's context (`lr_permutation`): log2 fold change of the group
medians of the depth-normalized scores against the pair's permutation p.
The outcome-associated component's top `N_TOP` pairs are highlighted and
named where p < 0.05, and its next `N_SHOWN - N_TOP` set off from the other
components' pairs."""

import numpy as np

from ....factors import load_factors_netcdf
from ... import common, style
from .._common import COMPONENT, SPEC
from ..coordinated_changes import N_TOP
from ..lr_permutation import CASE_DISPLAY, CONTROL_DISPLAY, permutation_analysis

N_SHOWN = 20
AXES = ("rect",)
# Labels: this far from their point (pt), moved to its left when another
# highlighted point lies within this x / y distance to its right
LABEL_OFFSET = 3
LABEL_CROWD_X = 1.0
LABEL_CROWD_Y = 0.15


def plot(axs):
    (ax,) = axs
    result = permutation_analysis()
    points = {
        k: (
            median_log2_fold_change(r)[r["order"][:N_SHOWN]],
            -np.log10(r["p"][r["order"][:N_SHOWN]]),
        )
        for k, r in result.items()
    }
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
    lr_names = load_factors_netcdf(SPEC.factors_path())["lr_pair"].values.astype(str)
    label_significant(
        ax, x[:N_TOP], y[:N_TOP], lr_names[result[COMPONENT]["order"][:N_TOP]]
    )
    ax.set(
        xlabel=f"log$_2$(median score ratio)\n{CASE_DISPLAY} ÷ {CONTROL_DISPLAY}",
        ylabel="−log$_{10}$(rank-difference p)",
    )
    common.legend(
        ax,
        "above",
        ncols=3,
        title=f"Top {N_SHOWN} LR pairs by loading",
        fontsize=style.FONT_SIZE_DENSE,
        title_fontsize=style.FONT_SIZE_DENSE,
    )


def median_log2_fold_change(component: dict) -> np.ndarray:
    """log2 ratio of the group medians, offset by a thousandth of the median
    positive one so pairs absent from a group stay finite."""
    case, control = component["medians"]
    both = np.concatenate([case, control])
    offset = 1e-3 * np.median(both[both > 0])
    return np.log2((case + offset) / (control + offset))


def label_significant(ax, x: np.ndarray, y: np.ndarray, names: np.ndarray) -> None:
    """Name the points above the p = 0.05 line: on their right, or on their
    left where another of these points crowds their right."""
    for xi, yi, name in zip(x, y, names, strict=True):
        if yi <= -np.log10(common.SIG_ALPHA):
            continue
        crowded = np.any(
            (x > xi) & (x - xi < LABEL_CROWD_X) & (np.abs(y - yi) < LABEL_CROWD_Y)
        )
        ax.annotate(
            name,
            (xi, yi),
            xytext=(-LABEL_OFFSET if crowded else LABEL_OFFSET, 0),
            textcoords="offset points",
            ha="right" if crowded else "left",
            va="center",
            fontsize=common.FACTOR_LABEL_FONT_SIZE,
        )


def makeFigure():
    return common.standalone_fixed(__name__)
