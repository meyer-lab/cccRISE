"""The NICHES panels (Figure 4, `figureS_niches_ibd`): cccRISE's IBD
sender and receiver populations against upstream NICHES, from the
`cccrise-niches` tables. A population's NICHES concordance is its share of
cells with a non-zero NICHES program score."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from ..comparators.niches_programs import (
    CONTRIBUTION,
    EXPRESSION,
    N_ORDERING_DRAWS,
    N_POPULATION_DRAWS,
    RANDOM,
    RANDOM_EXPRESSION,
    SIDES,
    WEIGHT,
    in_population,
)
from ..runners.datasets import NICHES_DIR, scores_filename
from . import common, style
from .figure2._common import GROUP_TAU, plot_weight_scatter

DATASET = "ibd"
#: Each figure's components, in drawing order
MAIN = (3, 4)
SUPPLEMENT = (1, 2, 5, 6)
ALL = tuple(sorted(MAIN + SUPPLEMENT))
#: Sides in shades of black: rows are labeled S and R, so color isn't needed
SIDE_SHADES = {"sender": "0.1", "receiver": "0.45"}
PROGRAM_COLORS = {
    1: style.CATEGORICAL[4],
    2: style.CATEGORICAL[2],
    3: style.COLORS["up"],
    4: style.COLORS["emphasis"],
    5: style.CATEGORICAL[5],
    6: style.CATEGORICAL[9],
}
SIDE_STYLES = {"sender": "-", "receiver": ":"}
NULL_COLOR = "0.88"
CONCORDANCE_LABEL = "NICHES concordance"
#: `orderings`: each block's header, then its (ordering, tick, x); the header
#: over every block, and the cohort column's
BLOCKS = {
    "State weight": ((RANDOM, "Random", 0.0), (WEIGHT, "CCC-RISE", 1.0)),
    "State weight × expression": (
        (RANDOM_EXPRESSION, "Random", 2.1),
        (CONTRIBUTION, "CCC-RISE", 3.1),
    ),
    "Expression": ((EXPRESSION, "Alone", 4.2),),
}
RANKING_HEADER = (
    "NICHES concordance of the top n cells (n = population size), ranked by"
)
COHORT_HEADER, COHORT_TICK = "Cohort", "% NICHES+"
SPARSITY_X = -1.1
#: `orderings`' column of each population's NICHES+ cells, left of the cohort's
COUNT_X, COUNT_HEADER, COUNT_TICK = -2.2, "NICHES+", "cells"
COUNT_NORM = LogNorm(10, 1000)
SPARSITY_CMAP = LinearSegmentedColormap.from_list(
    "sparsity", ["#fbeeec", style.COLORS["up"]]
)
SPARSITY_NORM = LogNorm(0.0005, 0.1)
#: Points between the S/R ticks and the outer component labels; and between
#: `orderings`' column ticks and block headers
OUTER_OFFSET = 11
HEADER_OFFSET = 10
RANKING_OFFSET = 21
#: `tau_sweep` draws a population only while it has this many cells
MIN_TAU_CELLS = 10
#: `weight_scatters`' ticks, on relative weights
SCATTER_TICKS = (0, 0.5, 1)
SCATTER_PAD = 0.05
#: `weight_scatters`' populations: a cell past both cutoffs is in both
POPULATION_COLORS = {
    "Sender": style.COLORS["sender"],
    "Receiver": style.COLORS["receiver"],
    "Both": style.COLORS["highlight"],
}


def load(name: str) -> pd.DataFrame:
    return common.load_runner_output(
        scores_filename(name, DATASET),
        NICHES_DIR,
        f"uv run cccrise-niches --dataset {DATASET}",
    )


def cases(components) -> list[tuple[int, str]]:
    return [(k, side) for k in components for side in SIDES]


def program_labels(ax, rows: list[tuple[int, str]], y: Sequence[float]) -> None:
    """Row labels as Figure 5 labels its fits: S and R at their own ticks, one
    outer component label per pair."""
    ax.set_yticks(
        y, [side[0].upper() for _, side in rows], fontsize=style.FONT_SIZE_DENSE
    )
    ax.tick_params(axis="y", pad=1.5, length=style.TICK_LENGTH)
    for i in range(0, len(rows), 2):
        ax.annotate(
            f"C{rows[i][0]}",
            xy=(0, (y[i] + y[i + 1]) / 2),
            xycoords=ax.get_yaxis_transform(),
            xytext=(-OUTER_OFFSET, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=style.FONT_SIZE_LABEL,
        )


def _concordance_row(ax, y: int, side: str, r: pd.Series) -> None:
    color = SIDE_SHADES[side]
    ax.barh(
        y,
        r["Direction 95th"] - r["Direction 5th"],
        left=r["Direction 5th"],
        height=0.62,
        color=NULL_COLOR,
        lw=0,
        zorder=1,
    )
    ax.plot(
        [r["Labels fine"], r["Observed"]],
        [y, y],
        color=color,
        lw=style.LINE_WIDTH,
        zorder=2,
    )
    ax.scatter(
        r["Labels coarse"],
        y,
        marker="|",
        s=22,
        color=style.COLORS["ink"],
        lw=0.8,
        zorder=3,
    )
    ax.scatter(
        r["Labels fine"],
        y,
        s=style.POINT_SIZE * 1.4,
        facecolor="white",
        edgecolor=color,
        lw=0.9,
        zorder=3,
    )
    ax.scatter(
        r["Observed"],
        y,
        marker="D",
        s=style.POINT_SIZE * 2,
        facecolor=color if r["Q direction"] < common.SIG_ALPHA else "white",
        edgecolor=color,
        lw=0.9,
        zorder=4,
    )


def _concordance_legend(ax, side: str | None) -> None:
    ink = style.COLORS["ink"]
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="D",
            ms=3.5,
            mfc=ink,
            mec=ink,
            label="CCC-RISE population",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=3.5,
            mfc="white",
            mec=ink,
            label="Fine label predicted",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="|",
            ms=5,
            mew=0.8,
            color=ink,
            label="Lineage predicted",
        ),
        Patch(facecolor=NULL_COLOR, label="Random directions"),
    ]
    spacing = {
        "handles": handles,
        "fontsize": style.FONT_SIZE_DENSE,
        "labelspacing": 0.2,
        "handletextpad": 0.3,
    }
    if side:
        common.legend(ax, side, **spacing)
    else:
        ax.legend(loc="lower right", borderaxespad=0.1, **spacing)


def concordance(axs, components, legend: str | None = None) -> None:
    """Each population of `components` and its NICHES concordance, against
    what its label composition predicts and random directions; the legend at
    the lower right, or outside the axes on side `legend`."""
    (ax,) = axs
    table = load("populations").set_index(["Component", "Side"]).loc[cases(components)]
    common.source_data(
        "niches_concordance_by_population",
        table.reset_index()
        .drop(columns=["Score observed", "Score labels fine", "Score fold"])
        .rename(
            columns={
                "Observed": "NICHES concordance",
                "Labels fine": "Fine label predicted",
                "Labels coarse": "Lineage predicted",
            }
        ),
        f"Each component's sender and receiver populations, the cells past "
        f"{GROUP_TAU} x that weight's largest value (a cell past both cutoffs is "
        "in both). NICHES concordance: share of the population's cells with a "
        "non-zero NICHES program score. Fine label and lineage predicted: the same "
        "with each cell at its type's rate. Direction 5th and 95th: percentiles of "
        f"the concordance of {N_POPULATION_DRAWS:,} random RISE directions, each "
        "taking as many cells from each fine type as the population; P and Q "
        "direction: one-sided p and BH q over all of the dataset's populations.",
    )
    rows = table.index.tolist()
    for y, (case, r) in enumerate(table.iterrows()):
        _concordance_row(ax, y, case[1], r)
    ax.set(xlim=(0, 1), ylim=(len(rows) - 0.5, -0.5), xlabel=CONCORDANCE_LABEL)
    ax.set_xticks(SCATTER_TICKS)
    program_labels(ax, rows, range(len(rows)))
    _concordance_legend(ax, legend)


def tau_sweep(axs, components) -> None:
    """Each population's concordance over its labels' across tau."""
    (ax,) = axs
    table = load("tau")
    table = table[table["Component"].isin(components)].assign(
        Gap=lambda t: t["Observed"] - t["Labels fine"]
    )
    common.source_data(
        "niches_concordance_by_tau",
        table.rename(
            columns={
                "Observed": "NICHES concordance",
                "Labels fine": "Fine label predicted",
                "Gap": "Concordance minus prediction",
            }
        ),
        "Each population at each tau (its cutoff as a fraction of that "
        "weight's largest value): its cells, NICHES concordance, what its fine "
        f"labels predict and the difference; {GROUP_TAU} is the figure's tau. "
        f"Points of fewer than {MIN_TAU_CELLS} cells are not drawn.",
    )
    ax.axhline(
        0,
        color=style.COLORS["reference"],
        lw=style.LINE_WIDTH_THIN,
        ls=style.LINE_STYLE_REFERENCE,
    )
    ax.axvline(
        GROUP_TAU, color=style.COLORS["muted"], lw=style.LINE_WIDTH_THIN, zorder=0
    )
    components = [k for k in components if k in set(table["Component"])]
    for k, side in cases(components):
        group = table[
            (table["Component"] == k)
            & (table["Side"] == side)
            & (table["Cells"] >= MIN_TAU_CELLS)
        ]
        ax.plot(
            group["Tau"],
            group["Gap"],
            color=PROGRAM_COLORS[k],
            lw=style.LINE_WIDTH,
            ls=SIDE_STYLES[side],
        )
    ax.set_xlabel("Tau (τ)")
    ax.set_ylabel("Δ(CCC-RISE, fine labels)", fontsize=style.FONT_SIZE_DENSE)
    handles = [
        Line2D([], [], color=PROGRAM_COLORS[k], lw=style.LINE_WIDTH, label=f"C{k}")
        for k in components
    ] + [
        Line2D(
            [],
            [],
            color=style.COLORS["ink"],
            lw=style.LINE_WIDTH,
            ls=ls,
            label=side.title(),
        )
        for side, ls in SIDE_STYLES.items()
    ]
    common.legend(ax, "right", handles=handles, fontsize=style.FONT_SIZE_DENSE)


def _share_text(share: float) -> str:
    return f"{share:.2%}" if share < 0.00095 else f"{share:.1%}"


def _block(ax, x0, x1, y, values, cmap, norm, text) -> None:
    """`values` (rows x columns) as one black-boxed block of cells."""
    box = (x0 - 0.5, x1 + 0.5, y[-1] + 0.5, y[0] - 0.5)
    ax.imshow(
        values, cmap=cmap, norm=norm, extent=box, aspect="auto", interpolation="nearest"
    )
    ax.add_patch(
        Rectangle(
            (box[0], box[3]),
            box[1] - box[0],
            box[2] - box[3],
            fill=False,
            edgecolor="black",
            lw=style.LINE_WIDTH_THIN,
            zorder=3,
        )
    )
    xs = np.linspace(x0, x1, values.shape[1])
    for (r, c), v in np.ndenumerate(values):
        ax.text(
            xs[c],
            y[r],
            text(v),
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            color="white" if norm(v) > 0.6 else style.COLORS["ink"],
        )


def _ordering_headers(ax) -> None:
    ticks = [(COUNT_X, COUNT_TICK), (SPARSITY_X, COHORT_TICK)] + [
        (x, name) for columns in BLOCKS.values() for _, name, x in columns
    ]
    ax.set_xticks(
        [x for x, _ in ticks], [n for _, n in ticks], fontsize=style.FONT_SIZE_DENSE
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=1.5)
    xs = [x for columns in BLOCKS.values() for _, _, x in columns]
    heads = [
        (COUNT_HEADER, COUNT_X, HEADER_OFFSET),
        (COHORT_HEADER, SPARSITY_X, HEADER_OFFSET),
    ] + [
        (name, (columns[0][2] + columns[-1][2]) / 2, HEADER_OFFSET)
        for name, columns in BLOCKS.items()
    ]
    heads.append((RANKING_HEADER, (xs[0] + xs[-1]) / 2, RANKING_OFFSET))
    for name, x, offset in heads:
        ax.annotate(
            name,
            xy=(x, 1),
            xycoords=ax.get_xaxis_transform(),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold",
            color=style.COLORS["ink"],
        )


def orderings(axs, components) -> None:
    """The population's number of cells chosen by a state direction, alone and
    times expression, and by expression alone; its NICHES+ cells, and how
    sparse NICHES is."""
    (ax,) = axs
    rows = cases(components)
    y = list(range(len(rows)))
    long = load("orderings")
    table = long.pivot(
        index=["Component", "Side"], columns="Ordering", values="Share"
    ).loc[rows]
    shares = long.groupby(["Component", "Side"])["Cohort share"].first().loc[rows]
    counts = (
        load("populations").set_index(["Component", "Side"]).loc[rows, "NICHES+ cells"]
    )
    common.source_data(
        "niches_concordance_by_ordering",
        table.join(shares).join(counts).reset_index(),
        "NICHES concordance of the population's number of cells, chosen by: a "
        f"random RISE direction (median of {N_ORDERING_DRAWS}); CCC-RISE's per-cell "
        "weight; the positive part of a random direction or the weight times the "
        "cell's centred expression of the component's ligands or receptors "
        "(Contribution); and that expression alone. Cohort share: NICHES+ cells' "
        "share of all cells. NICHES+ cells: the population's cells with a "
        "non-zero NICHES program score.",
    )
    linear = Normalize(0, 1)
    _block(
        ax,
        COUNT_X,
        COUNT_X,
        y,
        counts.to_numpy()[:, None],
        SPARSITY_CMAP,
        COUNT_NORM,
        lambda v: f"{v:,}",
    )
    _block(
        ax,
        SPARSITY_X,
        SPARSITY_X,
        y,
        shares.to_numpy()[:, None],
        SPARSITY_CMAP,
        SPARSITY_NORM,
        _share_text,
    )
    for columns in BLOCKS.values():
        names = [name for name, _, _ in columns]
        _block(
            ax,
            columns[0][2],
            columns[-1][2],
            y,
            table[names].to_numpy(),
            style.CMAPS["weight"],
            linear,
            lambda v: f"{v:.2f}",
        )
    _ordering_headers(ax)
    ax.set_xlim(COUNT_X - 0.5, 4.7)
    ax.set_ylim(y[-1] + 0.5, y[0] - 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    program_labels(ax, rows, y)


def weight_scatters(axs) -> None:
    """Each component's per-cell sender vs. receiver weight, relative to each
    weight's largest value, its populations (`in_population`) colored and the
    `GROUP_TAU` cutoffs drawn, on limits shared by every component."""
    adata = common.get_cell_projections(DATASET)
    senders, receivers = (
        np.asarray(adata.obsm[key]) / np.asarray(adata.obsm[key]).max(axis=0)
        for key in ("sc_B", "rc_C")
    )
    lower = min(senders.min(), receivers.min()) - SCATTER_PAD
    cells = {}
    for k, ax in enumerate(axs, start=1):
        sender, receiver = senders[:, k - 1], receivers[:, k - 1]
        sends, receives = in_population(sender), in_population(receiver)
        groups = {
            "Sender": sends & ~receives,
            "Receiver": receives & ~sends,
            "Both": sends & receives,
        }
        plot_weight_scatter(ax, sender, receiver, groups, POPULATION_COLORS)
        for line in (ax.axvline, ax.axhline):
            line(
                GROUP_TAU,
                color=style.COLORS["reference"],
                ls=style.LINE_STYLE_REFERENCE,
                lw=style.LINE_WIDTH_THIN,
            )
        ax.set(xlim=(lower, 1 + SCATTER_PAD), ylim=(lower, 1 + SCATTER_PAD))
        ax.set_title(f"C{k}", fontsize=style.FONT_SIZE_LABEL, pad=2)
        ax.set_xticks(SCATTER_TICKS)
        ax.set_yticks(SCATTER_TICKS, labels=None if k == 1 else [])
        ax.set_xlabel("Sender", fontsize=style.FONT_SIZE_DENSE)
        group = np.select(list(groups.values()), [g.lower() for g in groups], "rest")
        cells |= {
            f"Sender C{k}": sender,
            f"Receiver C{k}": receiver,
            f"Group C{k}": group,
        }
    axs[0].set_ylabel("Receiver", fontsize=style.FONT_SIZE_DENSE)
    axs[0].legend(
        handles=[
            Line2D([], [], ls="", marker="o", ms=2.5, mfc=c, mec=c, label=g.title())
            for g, c in POPULATION_COLORS.items()
        ],
        loc="upper left",
        fontsize=style.FONT_SIZE_DENSE,
        borderaxespad=0.2,
        handletextpad=0.1,
        labelspacing=0.2,
    )
    common.source_data(
        "cell_weights_groups",
        pd.DataFrame({"Cell": adata.obs_names, **cells}),
        "One row per cell: its sender and receiver weight on each CP component, "
        "each relative to that weight's largest value, and its group: sender or "
        f"receiver if only that relative weight is at least {GROUP_TAU}, both if "
        "both are, rest otherwise.",
    )
