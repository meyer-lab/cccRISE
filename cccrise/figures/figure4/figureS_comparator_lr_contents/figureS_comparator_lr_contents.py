"""Supplementary figure: what is in every comparator LR column the main
figure names (Px, Tx, Sx), coarse labels beside fine, one method per band.

Each component row is a stacked bar of its heaviest pairs' shares of its
squared LR loading (heaviest darkest; white is the rest) beside a dot matrix
of those pairs, one column per pair, each dot in its segment's shade (hollow
where the pair is signed against the column). Pairs are named a rank at a
time across a block's components while new ones fit its width, never one
under `MIN_SHARE` of a column's squared loading."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from ....comparators import scores
from ... import style
from .._common import COMPARATORS, METHOD_COLORS, METHOD_LABELS
from .._data import LABELLINGS, fit_lr, short

#: A pair under this share of a column's squared loading is never named.
MIN_SHARE = 0.01
#: Inches: component row height, pair column pitch, bar width, room for the
#: row names, gap between bar and dots, block title, and per character of
#: a rotated pair name.
ROW_IN, PITCH_IN, BAR_IN, NAME_IN, GAP_IN, TITLE_IN, CHAR_IN = (
    0.13,
    0.085,
    0.75,
    0.22,
    0.08,
    0.22,
    0.031,
)


def components(method: str, labelling: str) -> list[dict]:
    """Each LR column of `method`'s fit: its pairs' shares of squared
    loading (heaviest first) and signs, after orienting the column."""
    lr = fit_lr(method, labelling)
    rows = []
    for c in lr.columns:
        v = scores.oriented(lr[[c]])[c]
        share = (v**2 / (v**2).sum()).sort_values(ascending=False)
        rows.append(
            {
                "Name": short(method, c),
                "Pairs": list(share.index),
                "Shares": share.to_numpy(),
                "Signs": np.sign(v.loc[share.index]).to_numpy(),
            }
        )
    return rows


def _wanted(rows, depth, rank) -> list[int]:
    """The components still naming pairs whose `rank`-th pair clears
    `MIN_SHARE`, heaviest first."""
    keep = [
        i
        for i, r in enumerate(rows)
        if depth[i] == rank and r["Shares"][rank] >= MIN_SHARE
    ]
    return sorted(keep, key=lambda i: -rows[i]["Shares"][rank])


def _add_rank(rows, depth, columns, rank, max_columns) -> bool:
    """Name every component's `rank`-th pair that clears `MIN_SHARE`,
    heaviest first, while new pairs fit; whether any was wanted and all
    fit."""
    full = False
    wanted = _wanted(rows, depth, rank)
    for i in wanted:
        pair = rows[i]["Pairs"][rank]
        if pair not in columns:
            if len(columns) >= max_columns:
                full = True
                continue
            columns.append(pair)
        depth[i] += 1
    return bool(wanted) and not full


def chosen_pairs(rows, max_columns) -> tuple[list[int], list[str]]:
    """How many heaviest pairs each component names, and the block's pair
    columns in order of first use down the components."""
    depth, columns = [0] * len(rows), []
    for rank in range(max(len(r["Pairs"]) for r in rows)):
        if not _add_rank(rows, depth, columns, rank, max_columns):
            break
    first = {
        p: min(
            (i, r["Pairs"].index(p))
            for i, r in enumerate(rows)
            if p in r["Pairs"][: depth[i]]
        )
        for p in columns
    }
    return depth, sorted(columns, key=lambda p: first[p])


def shade(color, f):
    """`color` blended toward white, `f` = 1 the color itself."""
    return tuple(1 - f * (1 - x) for x in to_rgb(color))


def _draw_block(fig, rect, rows, depth, columns, color, title):
    """One fit's bar axes and dot axes at `rect` (x, y, total height in
    inches; figure size from `fig`)."""
    width, height = fig.get_size_inches()
    x0, y0, h = rect
    bar = fig.add_axes((x0 / width, y0 / height, BAR_IN / width, h / height))
    dots = fig.add_axes(
        (
            (x0 + BAR_IN + GAP_IN) / width,
            y0 / height,
            len(columns) * PITCH_IN / width,
            h / height,
        )
    )
    shades = [shade(color, f) for f in np.linspace(1.0, 0.4, max(max(depth), 2))]
    for y, (r, k) in enumerate(zip(rows, depth, strict=True)):
        lefts = np.concatenate([[0], np.cumsum(r["Shares"][:k])])
        for j in range(k):
            bar.barh(
                y,
                r["Shares"][j],
                left=lefts[j],
                height=0.74,
                color=shades[j],
                edgecolor=style.COLORS["ink"],
                lw=0.25,
            )
            positive = r["Signs"][j] > 0
            dots.scatter(
                columns.index(r["Pairs"][j]),
                y,
                s=11,
                facecolor=shades[j] if positive else "white",
                edgecolor=style.COLORS["ink"] if positive else shades[0],
                lw=0.3,
                zorder=3,
            )
        bar.barh(
            y,
            1 - lefts[k],
            left=lefts[k],
            height=0.74,
            color="white",
            edgecolor=style.COLORS["ink"],
            lw=0.25,
        )
    _style_block(bar, dots, rows, columns, title)


def _style_block(bar, dots, rows, columns, title):
    n = len(rows)
    dense = style.FONT_SIZE_DENSE - 1
    for ax in (bar, dots):
        ax.set_ylim(n - 0.5, -0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    bar.set_xlim(0, 1)
    bar.set_yticks(range(n), [r["Name"] for r in rows], fontsize=dense)
    bar.set_xticks((0, 0.5, 1), ["0", "0.5", "1"], fontsize=dense - 1)
    bar.set_xlabel("Squared loading", fontsize=dense - 0.5, labelpad=1)
    bar.tick_params(axis="y", length=0)
    bar.spines["left"].set_visible(False)
    bar.set_title(
        title, loc="left", fontsize=style.FONT_SIZE_DENSE, fontweight="bold", pad=3
    )
    dots.set_xlim(-0.6, len(columns) - 0.4)
    dots.set_xticks(
        range(len(columns)),
        [q.replace("^", "–") for q in columns],
        rotation=90,
        fontsize=dense - 1.5,
    )
    dots.set_yticks(range(n), [""] * n)
    dots.tick_params(length=0)
    dots.grid(color="0.9", lw=0.4)
    dots.set_axisbelow(True)


def draw_key(ax):
    """The key: dot fill for sign; equal shade squares with an arrow toward
    the heaviest; the white square for the rest."""
    ax.set(xlim=(0, 1.22), ylim=(0, 0.62), xticks=[], yticks=[])
    for side in ax.spines.values():
        side.set(visible=True, edgecolor=style.COLORS["ink"], lw=0.6)
    ink = style.COLORS["ink"]
    grey = [shade("0.25", f) for f in np.linspace(1.0, 0.4, 4)]
    text = {"fontsize": style.FONT_SIZE_DENSE - 1.5, "va": "center"}
    ax.scatter(0.12, 0.5, s=14, facecolor=grey[0], edgecolor=ink, lw=0.3)
    ax.text(0.2, 0.5, "+", **text)
    ax.scatter(0.42, 0.5, s=14, facecolor="white", edgecolor=grey[0], lw=0.5)
    ax.text(0.5, 0.5, "−", **text)
    side, y = 0.1, 0.27
    for k, color in enumerate(grey):
        ax.add_patch(
            Rectangle(
                (0.07 + k * side, y - side / 2),
                side,
                side,
                facecolor=color,
                edgecolor=ink,
                lw=0.3,
            )
        )
    ax.annotate(
        "",
        (0.07, 0.13),
        xytext=(0.07 + 4 * side, 0.13),
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.6,
            "color": ink,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    ax.text(0.07 + 4 * side + 0.04, 0.13, "heavier", **text)
    ax.add_patch(
        Rectangle(
            (0.66, y - side / 2), side, side, facecolor="white", edgecolor=ink, lw=0.4
        )
    )
    ax.text(0.66 + side + 0.04, y, "next LR pair\n<1% loaded", **text)


def _band(rows_by_labelling, chosen_by_labelling) -> float:
    """A method band's height (in): its longest block, title and the rotated
    pair names under it."""
    longest_name = max(len(q) for _, columns in chosen_by_labelling for q in columns)
    return (
        max(map(len, rows_by_labelling)) * ROW_IN
        + TITLE_IN
        + 0.14
        + CHAR_IN * longest_name
    )


def _block_left(method, labelling, coarse_columns, width) -> float:
    """A block's left edge (in): coarse at the left margin, fine at the
    middle, except the first method's fine block, tucked beside its coarse
    one to leave room for the key."""
    if labelling == "coarse":
        return NAME_IN
    if method == COMPARATORS[0]:
        return 2 * NAME_IN + BAR_IN + GAP_IN + coarse_columns * PITCH_IN + 0.3
    return width / 2 + NAME_IN


def makeFigure():
    width = style.FIGURE_WIDTH_DOUBLE
    max_columns = int((width / 2 - NAME_IN - BAR_IN - GAP_IN - 0.15) / PITCH_IN)
    blocks = {(m, lab): components(m, lab) for m in COMPARATORS for lab in LABELLINGS}
    chosen = {k: chosen_pairs(rows, max_columns) for k, rows in blocks.items()}
    bands = {
        m: _band(
            [blocks[(m, lab)] for lab in LABELLINGS],
            [chosen[(m, lab)] for lab in LABELLINGS],
        )
        for m in COMPARATORS
    }
    height = sum(bands.values()) + 0.1
    fig = plt.figure(figsize=(width, height))
    top = height - 0.1
    for m in COMPARATORS:
        for lab in LABELLINGS:
            x0 = _block_left(m, lab, len(chosen[(m, "coarse")][1]), width)
            h = len(blocks[(m, lab)]) * ROW_IN
            _draw_block(
                fig,
                (x0, top - TITLE_IN - h, h),
                blocks[(m, lab)],
                *chosen[(m, lab)],
                METHOD_COLORS[m],
                f"{METHOD_LABELS[m]}, {lab}",
            )
        top -= bands[m]
    key_width, key_height = 1.22, 0.62
    draw_key(
        fig.add_axes(
            (
                (width - key_width - 0.1) / width,
                (height - 0.3 - key_height) / height,
                key_width / width,
                key_height / height,
            )
        )
    )
    return fig
