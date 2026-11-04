"""The differential-detection volcano shared by this figure's sender vs.
receiver and group vs. rest panels: each gene's difference in the fraction of
cells detecting it against its Wilcoxon BH q (`S.rank_genes`), colored by
quadrant (`legend`) and with given genes labeled (`_label`)."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ... import common, style
from .. import _common as S

Q_FLOOR = 1e-30  # q below this drawn at it
HEADROOM = 1.1  # y limit over the highest point
LABEL_DX = (
    0.04  # labels start this far outward of their points, a fraction of the x range
)
LABEL_LINE_SPACING = 1.15  # between stacked labels, in font sizes
ROW_BAND = 1  # lines above or below a label whose points it starts beyond
REFERENCE_COLORS = {**S.GROUP_COLORS, "rest": style.COLORS["ink"]}
LABEL_BOX = {"edgecolor": "none", "pad": 0.5}  # pad in points
# Labels' boxes hide the reference lines under them, never a point
REFERENCE_ZORDER, LABEL_ZORDER = 0.5, 0.8  # points at 1
ARROWPROPS = {
    "arrowstyle": "-",
    "lw": style.LINE_WIDTH_THIN / 2,
    "color": style.COLORS["reference"],
}


def volcano_data(group: str, reference: str):
    """(Wilcoxon ranking, detected fraction difference, -log10 q) of `group`
    against `reference` ('receiver', or 'rest': every cell outside `group`)."""
    if reference == "rest":  # scanpy's "rest": every other cell; one cached test
        ranking = S.rank_genes(S.DE_GROUPS, "rest")[group]
    else:
        ranking = S.rank_genes((group,), reference)[group]
    in_group, in_reference = detected_fractions(group, reference, ranking.index)
    delta = in_group - in_reference
    return ranking, delta, -np.log10(np.clip(ranking["q"], Q_FLOOR, 1))


def detected_fractions(group: str, reference: str, genes):
    """Fraction of `group`'s and of `reference`'s cells detecting each of
    `genes`, 'rest' pooling every cell outside `group`."""
    detected, _ = S.group_expression(list(genes))
    if reference != "rest":
        return detected[group], detected[reference]
    sizes = S.cell_groups().value_counts()
    others = [g for g in S.GROUPS if g != group]
    pooled = detected[others] @ sizes[others] / sizes[others].sum()
    return detected[group], pooled


def volcano_table(group: str, reference: str) -> pd.DataFrame:
    """Every gene's point on the volcano of `group` against `reference`, with
    its Wilcoxon statistics and detected fractions, as source data."""
    ranking, delta, y = volcano_data(group, reference)
    in_group, in_reference = detected_fractions(group, reference, ranking.index)
    return pd.DataFrame(
        {
            "Gene": ranking.index,
            "Wilcoxon score": ranking["score"].to_numpy(),
            "p": ranking["p"].to_numpy(),
            "q (BH)": ranking["q"].to_numpy(),
            f"Detected fraction, {group}": in_group.to_numpy(),
            f"Detected fraction, {name(reference)}": in_reference.to_numpy(),
            "Detected fraction difference": delta.to_numpy(),
            f"-log10 q (plotted, floored at q = {Q_FLOOR:g})": y.to_numpy(),
            "Enriched": enriched(ranking, delta).to_numpy(),
        }
    )


def save_volcano(group: str, reference: str, part: int | None = None) -> None:
    """`volcano_table` of `group` against `reference` as the panel's source data."""
    common.source_data(
        f"de_{group}_vs_{reference}",
        volcano_table(group, reference),
        f"One row per gene: two-sided Wilcoxon rank-sum test (scanpy) of the "
        f"{group} group's cells against the {name(reference)} cells' normalized "
        "expression, p-values Benjamini-Hochberg corrected across genes (q); the "
        "fraction of each group's cells detecting the gene and their difference "
        "(x axis); 'Enriched' marks a gene with q < 0.05 whose Wilcoxon score has "
        "the same sign as its detected fraction difference (colored).",
        part=part,
    )


def name(group: str) -> str:
    """`group` as the volcanoes display it: 'rest' is every other cell."""
    return "other" if group == "rest" else group


def enriched(ranking, delta):
    """Whether each gene is significant with its Wilcoxon score on the same
    side of zero as its detected fraction difference."""
    return (ranking["q"] < common.SIG_ALPHA) & (
        np.sign(ranking["score"]) == np.sign(delta)
    )


def limits(delta, y) -> tuple[float, float]:
    """(symmetric x limit, y limit) fitting `delta` and `y`, with label room."""
    return np.abs(delta).max() * 1.25, y.max() * HEADROOM


def plot_volcano(ax, group: str, reference: str, labeled, lims=None) -> None:
    """Volcano of `group` against `reference` within `lims` (default `limits`
    of its own data), labeling `labeled` (genes, or {gene: label text}). A
    significant gene takes the color of the side of zero it falls on, unless
    its Wilcoxon score points the other way (ubiquitous genes, e.g.
    mitochondrial, detected slightly less often but more highly), which stays
    muted."""
    ranking, delta, y = volcano_data(group, reference)
    color = np.where(
        enriched(ranking, delta),
        np.where(delta > 0, REFERENCE_COLORS[group], REFERENCE_COLORS[reference]),
        style.COLORS["muted"],
    )
    ax.scatter(
        delta, y, c=color, s=style.POINT_SIZE_BACKGROUND, linewidths=0, rasterized=True
    )
    for line in (
        common.reference_line(ax, -np.log10(common.SIG_ALPHA)),
        common.reference_line(ax, 0, horizontal=False),
    ):
        line.set_zorder(REFERENCE_ZORDER)
    x_limit, y_limit = limits(delta, y) if lims is None else lims
    ax.set(xlim=(-x_limit, x_limit), ylim=(0, y_limit))
    _label(
        ax, labeled if isinstance(labeled, dict) else {g: g for g in labeled}, delta, y
    )
    ax.set(
        xlabel=f"Detected fraction,\n{group} − {name(reference)}",
        ylabel="−log$_{10}$ q (Wilcoxon)",
    )


def legend(ax) -> None:
    """The volcanoes' shared color key, its title over one row of entries,
    above `ax` (below its panel letter)."""
    entries = {name(g).capitalize(): c for g, c in REFERENCE_COLORS.items()}
    entries["None"] = style.COLORS["muted"]
    handles = [
        Line2D([], [], ls="", marker="o", color=c, label=label)
        for label, c in entries.items()
    ]
    legend = common.legend(
        ax, "above", handles=handles, ncol=len(handles), title="Enriched in"
    )
    # Wider than the axes: hangs into the row's gaps rather than spreading the
    # panels apart, its height kept by an empty title padded past it
    legend.set_in_layout(False)
    height = legend.get_window_extent().height / ax.figure.dpi + style.LEGEND_PAD
    ax.set_title(" ", pad=height * 72)


def _label(ax, labels: dict[str, str], delta, y) -> None:
    """Labels (`labels`, {gene: text}) of genes, stacked at least a line apart
    on each side of zero by height (`_stack`), each just outward of every
    point within `ROW_BAND` lines of it (away from zero, where the volcano is
    empty), with leader lines to their points; each on a background-colored
    box, hiding the reference lines under it."""
    step = (
        LABEL_LINE_SPACING
        * common.FACTOR_LABEL_FONT_SIZE
        / 72
        * ax.figure.dpi
        / ax.bbox.height
        * np.diff(ax.get_ylim())[0]
    )
    dx = LABEL_DX * np.diff(ax.get_xlim())[0]
    for side in (1, -1):
        on_side = sorted(
            (g for g in labels if (delta[g] >= 0) == (side > 0)), key=lambda g: -y[g]
        )
        for g, slot in zip(on_side, _stack(y[on_side].to_numpy(), step), strict=True):
            ax.annotate(
                labels[g],
                xy=(delta[g], y[g]),
                xytext=(side * (_row_edge(delta * side, y, slot, step) + dx), slot),
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=common.FACTOR_LABEL_FONT_SIZE,
                bbox=LABEL_BOX | {"facecolor": ax.get_facecolor()},
                zorder=LABEL_ZORDER,
                arrowprops=ARROWPROPS,
                annotation_clip=False,
            )


def _row_edge(outward, y, slot: float, step: float) -> float:
    """The farthest outward position (`outward`, the detected fraction
    difference signed so the label's side is positive) of the points within
    `ROW_BAND` lines of `slot`, or zero."""
    return np.max(outward[np.abs(y - slot) < ROW_BAND * step].to_numpy(), initial=0)


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
