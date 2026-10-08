"""The differential-detection volcano shared by this figure's sender vs.
receiver and group vs. rest panels: each gene's difference in the fraction of
cells detecting it against its Wilcoxon BH q (`S.rank_genes`). The
`N_LABELED` most enriched significant genes on each side (`N_LABELED_REST`
on the rest's) are labeled, the
genes of the top LR pairs (`S.top_pair_genes`) outlined (and labeled, bold, if
significant), and the genes both groups enrich against the rest
(`shared_genes`) colored `SHARED_COLOR` and labeled on every panel. Labels
are placed by `_label`."""

import functools

import numpy as np
from adjustText import adjust_text

from ... import common, style
from .. import _common as S

N_LABELED = 6
N_LABELED_REST = 3  # rest-enriched genes labeled: immune genes absent in both groups
N_SHARED = 10  # top enriched genes per group compared for `shared_genes`
Q_FLOOR = 1e-30  # q below this drawn at it
HEADROOM = 1.5  # y limit over the highest point, room for labels
SHARED_COLOR = style.COLORS["highlight"]
LABEL_DX = (
    0.04  # labels start this far outward of their points, a fraction of the x range
)
LABEL_LINE_SPACING = 1.15  # between stacked labels, in font sizes
REFERENCE_COLORS = {**S.GROUP_COLORS, "rest": style.COLORS["ink"]}
ARROWPROPS = {
    "arrowstyle": "-",
    "lw": style.LINE_WIDTH_THIN / 2,
    "color": style.COLORS["reference"],
}


@functools.cache
def shared_genes() -> list[str]:
    """Genes among the `N_SHARED` most enriched significant genes of both the
    sender and receiver groups against the rest."""
    ranked = S.rank_genes(S.DE_GROUPS, "rest")
    top = [
        set(r.index[(r["q"] < common.SIG_ALPHA) & (r["score"] > 0)][:N_SHARED])
        for r in ranked.values()
    ]
    return sorted(set.intersection(*top))


def volcano_data(group: str, reference: str):
    """(Wilcoxon ranking, detected fraction difference, -log10 q) of `group`
    against `reference` ('receiver' or 'rest')."""
    if reference == "rest":  # shares `shared_genes`' cached test
        ranking = S.rank_genes(S.DE_GROUPS, "rest")[group]
    else:
        ranking = S.rank_genes((group,), reference)[group]
    detected, _ = S.group_expression(ranking.index.tolist())
    delta = detected[group] - detected[reference]
    return ranking, delta, -np.log10(np.clip(ranking["q"], Q_FLOOR, 1))


def limits(delta, y) -> tuple[float, float]:
    """(symmetric x limit, y limit) fitting `delta` and `y`, with label room."""
    return np.abs(delta).max() * 1.25, y.max() * HEADROOM


def plot_volcano(ax, group: str, reference: str, lims=None) -> None:
    """Volcano of `group` against `reference`, genes enriched in each colored
    by it if significant, within `lims` (default `limits` of its own data)."""
    ranking, delta, y = volcano_data(group, reference)
    significant = ranking.index[ranking["q"] < common.SIG_ALPHA]
    color = np.where(
        ranking["q"] < common.SIG_ALPHA,
        np.where(
            ranking["score"] > 0, REFERENCE_COLORS[group], REFERENCE_COLORS[reference]
        ),
        style.COLORS["muted"],
    )
    ax.scatter(
        delta, y, c=color, s=style.POINT_SIZE_BACKGROUND, linewidths=0, rasterized=True
    )
    lr, shared = S.top_pair_genes(), shared_genes()
    ax.scatter(
        delta[lr],
        y[lr],
        facecolors="none",
        edgecolors=style.COLORS["ink"],
        s=style.POINT_SIZE_FOREGROUND,
        linewidths=style.LINE_WIDTH_THIN,
    )
    ax.scatter(
        delta[shared], y[shared], c=SHARED_COLOR, s=style.POINT_SIZE, linewidths=0
    )
    common.reference_line(ax, -np.log10(common.SIG_ALPHA))
    common.reference_line(ax, 0, horizontal=False)
    x_limit, y_limit = limits(delta, y) if lims is None else lims
    ax.set(xlim=(-x_limit, x_limit), ylim=(0, y_limit))
    labeled = [
        *significant[:N_LABELED],
        *significant[::-1][
            : N_LABELED if reference in S.GROUP_COLORS else N_LABELED_REST
        ],
        *(g for g in lr if g in significant),
        *shared,
    ]
    _label(ax, list(dict.fromkeys(labeled)), delta, y)
    ax.set(
        xlabel=f"Detected fraction,\n{group} − {reference}",
        ylabel="−log$_{10}$ q (Wilcoxon)",
    )


def _label(ax, genes, delta, y) -> None:
    """Labels of `genes`, first placed just outward of their points (away
    from zero, where the volcano is empty) and stacked at least a line apart
    on each side (coincident points otherwise trap `adjust_text`), then moved
    off the points by `adjust_text`, with leader lines to their points."""
    step = (
        LABEL_LINE_SPACING
        * common.FACTOR_LABEL_FONT_SIZE
        / 72
        * ax.figure.dpi
        / ax.bbox.height
        * np.diff(ax.get_ylim())[0]
    )
    lr, shared = S.top_pair_genes(), shared_genes()
    texts = []
    for side in (1, -1):
        on_side = sorted(
            (g for g in genes if np.sign(delta[g]) == side), key=lambda g: -y[g]
        )
        for g, slot in zip(on_side, _stack(y[on_side].to_numpy(), step), strict=True):
            texts.append(
                ax.text(
                    delta[g] + side * LABEL_DX * np.diff(ax.get_xlim())[0],
                    slot,
                    g,
                    ha="left" if side > 0 else "right",
                    va="center",
                    fontsize=common.FACTOR_LABEL_FONT_SIZE,
                    fontweight="bold" if g in lr else "normal",
                    color=SHARED_COLOR if g in shared else None,
                )
            )
    placed = [t.get_text() for t in texts]
    adjust_text(
        texts,
        x=delta.to_numpy(),
        y=y.to_numpy(),
        target_x=delta[placed].to_numpy(),
        target_y=y[placed].to_numpy(),
        ax=ax,
        expand=(1.2, 1.4),
        force_text=(0.6, 1.0),
        force_pull=(0.002, 0.002),
        time_lim=10,
        arrowprops=ARROWPROPS,
    )


def _stack(heights: np.ndarray, step: float) -> list[float]:
    """Positions for labels at descending `heights`, each at its height but at
    least `step` below the one above and above the x axis."""
    slots = []
    for h in heights:
        slots.append(min(h, slots[-1] - step) if slots else h)
    for i in reversed(range(len(slots))):
        floor = step / 2 if i == len(slots) - 1 else slots[i + 1] + step
        slots[i] = max(slots[i], floor)
    return slots
