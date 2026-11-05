"""The NICHES panels (`figure5_niches`, `figureS_niches_ibd`,
`figureS_niches_alad`): cccRISE's sender and receiver populations against
upstream NICHES, from the `cccrise-niches` tables. A population's NICHES
concordance is its share of cells with a non-zero NICHES program score."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from ..comparators.niches_programs import (
    CONTRIBUTION,
    EXPRESSION,
    N_ORDERING_DRAWS,
    N_POPULATION_DRAWS,
    RANDOM,
    RANDOM_EXPRESSION,
    SCORABLE,
    SIDES,
    WEIGHT,
)
from ..runners.datasets import NICHES_DIR, scores_filename
from . import common, style
from .figure2._common import GROUP_TAU

#: Each figure's components, in the groups drawn together
IBD_MAIN = ((3, 4),)
IBD_SUPPLEMENT = ((1, 2, 5, 6),)
ALAD = ((1, 5), (2, 3, 4))
#: Sides in shades of black: rows are labeled S and R, so color isn't needed
SIDE_SHADES = {"sender": "0.1", "receiver": "0.45"}
PROGRAM_COLORS = {
    1: style.CATEGORICAL[4],
    2: style.CATEGORICAL[2],
    3: style.COLORS["up"],
    4: style.COLORS["emphasis"],
    5: style.CATEGORICAL[5],
}
SIDE_STYLES = {"sender": "-", "receiver": ":"}
NULL_COLOR = "0.88"
CONCORDANCE_LABEL = "NICHES concordance"
#: The gap between `concordance`'s stacked axes
STACK_GAP = 0.08
#: `orderings`: each block's header, then its (ordering, tick, x)
BLOCKS = {
    "State": ((RANDOM, "Random", 0.0), (WEIGHT, "cccRISE", 1.0)),
    "State × Expression": (
        (RANDOM_EXPRESSION, "Random", 2.1),
        (CONTRIBUTION, "cccRISE", 3.1),
    ),
    "Expression": ((EXPRESSION, "Alone", 4.2),),
}
SPARSITY_X = -1.1
SPARSITY_CMAP = LinearSegmentedColormap.from_list(
    "sparsity", ["#fbeeec", style.COLORS["up"]]
)
SPARSITY_NORM = LogNorm(0.0005, 0.1)
#: Row gap between a family of components in `orderings` and the next
FAMILY_GAP = 0.5
#: Points between the S/R ticks and the outer component labels; and between
#: `orderings`' column ticks and block headers
OUTER_OFFSET = 11
HEADER_OFFSET = 10
#: `tau_sweep` draws a population only while it has this many cells
MIN_TAU_CELLS = 10
#: Extra room between the figures' two rows
ROW_GAP = 6 * style.MM


def load(name: str, dataset: str) -> pd.DataFrame:
    return common.load_runner_output(
        scores_filename(name, dataset),
        NICHES_DIR,
        f"uv run cccrise-niches --dataset {dataset}",
    )


def compose(layout: list[list]) -> Figure:
    """`layout` at full width, its rows `ROW_GAP` apart."""
    gap, style.ROW_GAP = style.ROW_GAP, ROW_GAP
    try:
        return common.compose_fixed(layout, width=style.FIGURE_WIDTH_DOUBLE)
    finally:
        style.ROW_GAP = gap


def cases(components) -> list[tuple[int, str]]:
    return [(k, side) for k in components for side in SIDES]


def program_labels(ax, rows: list[tuple[int, str]], y: Sequence[float]) -> None:
    """Row labels as Figure 4 labels its fits: S and R at their own ticks, one
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


def _scorable_groups(table: pd.DataFrame, groups) -> list[list[tuple[int, str]]]:
    """`groups`' populations that are scorable, a group of none dropped."""
    scorable = table[table["Scorable"]].index
    rows = [[c for c in cases(g) if c in scorable] for g in groups]
    return [r for r in rows if r]


def _stacked_axes(ax, sizes: list[int]) -> list:
    """Axes stacked inside `ax`, each as tall as its number of rows."""
    row = (1 - STACK_GAP * (len(sizes) - 1)) / sum(sizes)
    stacks, top = [], 1.0
    for size in sizes:
        stacks.append(ax.inset_axes((0, top - size * row, 1, size * row)))
        top -= size * row + STACK_GAP
    return stacks


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


def _concordance_legend(ax, outside: bool) -> None:
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
            label="cccRISE population",
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
    if outside:
        common.legend(ax, "right", **spacing)
    else:
        ax.legend(loc="lower right", borderaxespad=0.1, **spacing)


def concordance(axs, dataset: str, groups, legend_stack: int | None = -1) -> None:
    """Each scorable population of `groups` and its NICHES concordance, against
    what its label composition predicts and random directions; the legend in
    stacked axes `legend_stack`, clear of the data, or right of the last if
    None."""
    (ax,) = axs
    table = load("populations", dataset).set_index(["Component", "Side"])
    table = table.loc[[c for g in groups for c in cases(g)]]
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
        f"{GROUP_TAU} x that weight's largest value, split by relative weight "
        "(Figure 2). NICHES concordance: share of the population's cells with a "
        "non-zero NICHES program score. Fine label and lineage predicted: the same "
        "with each cell at its type's rate. Direction 5th and 95th: percentiles of "
        f"the concordance of {N_POPULATION_DRAWS:,} random RISE directions, each "
        "taking as many cells from each fine type as the population; P and Q "
        "direction: one-sided p and BH q over the dataset's scorable populations. "
        f"Scorable: at least {SCORABLE:.0%} of the population is NICHES+.",
    )
    rows_by_stack = _scorable_groups(table, groups)
    ax.set_axis_off()
    stacks = _stacked_axes(ax, [len(g) for g in rows_by_stack])
    for stack, rows in zip(stacks, rows_by_stack, strict=True):
        for y, (case, r) in enumerate(table.loc[rows].iterrows()):
            _concordance_row(stack, y, case[1], r)
        stack.set(xlim=(0, 1), ylim=(len(rows) - 0.5, -0.5))
        program_labels(stack, rows, range(len(rows)))
        stack.tick_params(axis="x", labelbottom=stack is stacks[-1])
    stacks[-1].set_xlabel(CONCORDANCE_LABEL)
    if legend_stack is None:
        _concordance_legend(stacks[-1], outside=True)
    else:
        _concordance_legend(stacks[legend_stack], outside=False)


def tau_sweep(axs, dataset: str, groups) -> None:
    """Each scorable population's concordance over its labels' across tau."""
    (ax,) = axs
    components = [k for g in groups for k in g]
    table = load("tau", dataset)
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
        "Each scorable population at each tau (its cutoff as a fraction of that "
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
    ax.set_ylabel("Δ(cccRISE, cell type prediction)", fontsize=style.FONT_SIZE_DENSE)
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


def _family_rows(
    groups,
) -> tuple[list[tuple[int, str]], list[float], list[slice]]:
    """Every component and side, each one's y, and each group's slice of them."""
    rows, y, families, top = [], [], [], 0.0
    for group in groups:
        family = cases(group)
        families.append(slice(len(rows), len(rows) + len(family)))
        rows += family
        y += [top + i for i in range(len(family))]
        top = y[-1] + 1 + FAMILY_GAP
    return rows, y, families


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
    ticks = [(SPARSITY_X, "% of cells")] + [
        (x, name) for columns in BLOCKS.values() for _, name, x in columns
    ]
    ax.set_xticks(
        [x for x, _ in ticks], [n for _, n in ticks], fontsize=style.FONT_SIZE_DENSE
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=1.5)
    heads = {"NICHES+": SPARSITY_X} | {
        name: (columns[0][2] + columns[-1][2]) / 2 for name, columns in BLOCKS.items()
    }
    for name, x in heads.items():
        ax.annotate(
            name,
            xy=(x, 1),
            xycoords=ax.get_xaxis_transform(),
            xytext=(0, HEADER_OFFSET),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold",
            color=style.COLORS["ink"],
        )


def orderings(axs, dataset: str, groups) -> None:
    """The population's number of cells chosen by a state direction, alone and
    times expression, and by expression alone; and how sparse NICHES is."""
    (ax,) = axs
    rows, y, families = _family_rows(groups)
    long = load("orderings", dataset)
    table = long.pivot(
        index=["Component", "Side"], columns="Ordering", values="Share"
    ).loc[rows]
    shares = long.groupby(["Component", "Side"])["Cohort share"].first().loc[rows]
    common.source_data(
        "niches_concordance_by_ordering",
        table.join(shares).reset_index(),
        "NICHES concordance of the population's number of cells, chosen by: a "
        f"random RISE direction (median of {N_ORDERING_DRAWS}); cccRISE's per-cell "
        "weight; the positive part of a random direction or the weight times the "
        "cell's centred expression of the component's ligands or receptors "
        "(Contribution); and that expression alone. Cohort share: NICHES+ cells' "
        "share of all cells.",
    )
    linear = Normalize(0, 1)
    sparsity = shares.to_numpy()[:, None]
    for family in families:
        _block(
            ax,
            SPARSITY_X,
            SPARSITY_X,
            y[family],
            sparsity[family],
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
                y[family],
                table[names].to_numpy()[family],
                style.CMAPS["weight"],
                linear,
                lambda v: f"{v:.2f}",
            )
    _ordering_headers(ax)
    ax.set_xlim(SPARSITY_X - 0.5, 4.7)
    ax.set_ylim(y[-1] + 0.5, y[0] - 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    program_labels(ax, rows, y)
