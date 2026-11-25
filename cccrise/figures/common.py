"""
Functions shared across manuscript figures: data loading, panel layout, the
build entry point, and plotting functions used by more than one figure (e.g.
the same rank-selection panels for ALAD and IBD). Code used by a single panel
lives in that panel's module instead, and settings, data, or functions shared
only by one main figure and its supplements live in that main figure's
private `_common.py`.

Each figure lives in its own `figures/figure<N>/` package: one module per
panel, each exposing `plot(fig)` (draws into a Figure or SubFigure) and
`makeFigure()` (the panel alone), plus a `figure<N>.py` master that arranges
the panels with `panel_grid` and labels them with `label_panels`. Each
supplementary figure is a `figureS_<name>/` package nested in the main
figure it supports, or at the top level if it supports no single one.

Each panel saves the values it plots with `source_data` (or declares
`SCHEMATIC = True`), which `figures/source_data.py` packages as EMBO source
data.
"""

import functools
import importlib
import itertools
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import anndata
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure, SubFigure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.offsetbox import DrawingArea
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import blended_transform_factory
from matplotlib.typing import LegendLocType

matplotlib.use("AGG")

from ..anova import (
    get_sample_to_group,
    prepare_group_labels,
)
from ..factors import (
    ALAD_FACTORS_PATH,
    DATA_DIR,
    LOADED_PATHS,
    clean_memory,
    load_factors_to_anndata,
)
from ..import_data import import_alad
from ..rank_selection import distinct_minima
from ..runners.datasets import (
    ALAD_PROGRAMS_DIR,
    DATASETS,
    RANK_SELECTION_DIR,
    scores_filename,
)
from ..runners.provenance import read_metadata
from ..utils import (
    correct_conditions,
)
from . import style

__all__ = [
    "ALAD_FACTORS_PATH",
    "DATA_DIR",
    "clean_memory",
    "genFigure",
    "get_alad_data",
    "load_factors_to_anndata",
    "load_runner_output",
]


def get_alad_data(
    project_cells: bool = False, do_embedding: bool = False
) -> anndata.AnnData:
    """Load ALAD dataset with factors and optional single-cell projections."""
    if not project_cells:
        return load_factors_to_anndata(ALAD_FACTORS_PATH)
    raw = import_alad(gene_threshold=0.001, normalize=True)
    return load_factors_to_anndata(
        ALAD_FACTORS_PATH, adata=raw, do_embedding=do_embedding
    )


def load_runner_output(
    filename: str, root: Path, hint: str | None = None
) -> pd.DataFrame:
    """Read a sweep result written by one of the runner scripts.

    Figures plot what a runner already produced; they never compute a sweep
    themselves, so a missing file is an error naming the command to run.

    Parameters
    ----------
    filename : str
        The runner's output filename.
    root : Path
        Directory the file lives in -- the runner's analysis directory
        (`runners.datasets.RANK_SELECTION_DIR` / `VALIDATION_DIR`) for a
        default-settings output, or a `runners.datasets.comparison_output_dir`
        for an alternate-score one.
    hint : str, optional
        Command to suggest in the error when the file is missing. Defaults
        to the rank-sweep runners' commands.
    """
    path = root / filename
    if not path.exists():
        hint = hint or (
            "    uv run cccrise-riserank --dataset <dataset> --ranks 1:51\n"
            "    uv run cccrise-cpdrank  --dataset <dataset>\n"
            "    uv run cccrise-stability --dataset <dataset> --rise-rank <N> --cp-rank <K>"
        )
        raise FileNotFoundError(
            f"{path} not found. Produce it with the matching runner, e.g.\n{hint}"
        )
    _SOURCE["inputs"].add(path.resolve())
    return pd.read_parquet(path)


@functools.cache
def alad_programs(table: str) -> pd.DataFrame:
    """A `cccrise-alad-programs` table (`run_alad_programs`): Figure 2's
    component 5 group tests and Figure 3's program tests. Shared; don't
    modify it."""
    return load_runner_output(
        scores_filename(table, "alad"),
        ALAD_PROGRAMS_DIR,
        "    uv run cccrise-alad-programs --dataset alad",
    )


def load_cpd_rank(dataset: str) -> pd.DataFrame:
    """`cccrise-cpdrank`'s sweep for `dataset`: one row per CPD rank and
    seeded start (see `production_fits`)."""
    return load_runner_output(DATASETS[dataset].cpd_rank_filename(), RANK_SELECTION_DIR)


def production_fits(sweep: pd.DataFrame) -> pd.DataFrame:
    """The production start's row (start 0) at each rank of a `load_cpd_rank`
    sweep."""
    return sweep[sweep["Start"] == 0]


def load_cpd_ridge(dataset: str) -> pd.DataFrame:
    """`cccrise-cpdrank`'s ridge comparison for `dataset`: one row per (rank,
    component), with per-rank columns repeated (see `per_rank`)."""
    return load_runner_output(
        DATASETS[dataset].cpd_ridge_filename(), RANK_SELECTION_DIR
    )


def load_cpd_imputation(dataset: str) -> pd.DataFrame:
    """`cccrise-cpdrank`'s imputation R2X for `dataset`: one row per (rank,
    fold split)."""
    return load_runner_output(
        DATASETS[dataset].cpd_imputation_filename(), RANK_SELECTION_DIR
    )


def per_rank(sweep: pd.DataFrame) -> pd.DataFrame:
    """One row per rank of a `load_cpd_ridge` table."""
    return sweep.drop_duplicates("Rank")


def selected_ranks(dataset: str) -> tuple[int, int]:
    """(RISE rank, CPD rank) of `dataset`'s canonical fit, from its provenance."""
    ranks = read_metadata(DATASETS[dataset].factors_path())["rank_selection"]
    return ranks["rise"]["rank"], ranks["cpd"]["rank"]


# =============================================================================
# Layout and build
# =============================================================================

FIGURES_DIR = Path(__file__).parent
OUTPUT_DIR = FIGURES_DIR.parent.parent / "output"
# Panel letters of every built figure, read by the manuscript's `meta` shortcodes
PANELS_PATH = FIGURES_DIR.parent.parent / "manuscript" / "_panels.yml"


def panel_grid(
    figsize: tuple[float, float], layout: list[list[str]], **gridspec_kw
) -> tuple[Figure, dict[str, SubFigure]]:
    """A figure divided into one subfigure per panel.

    `layout` is a mosaic: each row names the panel occupying each grid cell,
    and a panel named in several adjacent cells spans them. Panels are
    returned in reading order, the order `label_panels` letters them in.
    """
    fig = plt.figure(figsize=figsize, layout="constrained")
    grid = fig.add_gridspec(len(layout), len(layout[0]), **gridspec_kw)
    cells = np.array(layout)
    panels = {}
    for name in dict.fromkeys(cells.ravel()):
        rows, cols = np.nonzero(cells == name)
        panels[name] = fig.add_subfigure(
            grid[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        )
    return fig, panels


def label_panels(panels: dict[str, SubFigure]) -> None:
    """Label panels A, B, C, ... in order, at each panel's top left, tagged
    with the panel's name for `record_panel_letters`."""
    for letter, (name, panel) in zip(ascii_uppercase, panels.items(), strict=False):
        panel.suptitle(letter, x=0, ha="left", gid=name)


def compose(
    figsize: tuple[float, float], layout: list[list[str]], panels: dict, **gridspec_kw
) -> Figure:
    """A master figure: each panel module in `panels` (name -> module with a
    `plot(fig)`) drawn into its `layout` cells, then lettered."""
    fig, subfigs = panel_grid(figsize, layout, **gridspec_kw)
    for name, subfig in subfigs.items():
        _SOURCE["panel"] = panels[name].__name__.rpartition(".")[2]
        panels[name].plot(subfig)
    label_panels(
        {panels[name].__name__.rpartition(".")[2]: sf for name, sf in subfigs.items()}
    )
    return fig


def standalone(
    plot: Callable[[Figure | SubFigure], None],
    size: tuple[float, float],
    legend: Callable[[Figure], None] | None = None,
) -> Figure:
    """One panel drawn on a figure of its own, with `legend(fig)` for panels
    whose master figure shares one legend across panels."""
    fig = plt.figure(figsize=size, layout="constrained")
    _SOURCE["panel"] = plot.__module__.rpartition(".")[2]
    plot(fig)
    if legend is not None:
        legend(fig)
    return fig


def compose_fixed(rows: list[list], width: float | None = None) -> Figure:
    """A figure of fixed-size axes, each lettered at its top left.

    Each row lists panel modules, each declaring its axes' `style.AXIS_SIZE`
    kinds in `AXES` and drawing into them with `plot(axs)`; a list of modules
    in a row is stacked top to bottom in one slot (`_Stack`). Axes are drawn at
    their final size, then each one's margins (ticks, labels, legends, inset
    colorbars) are measured and set the space around it, so they never shrink
    the axes. Rows are centered, spreading leftover width into the gaps up
    to `style.PANEL_GAP_MAX`; `width` defaults to the widest row's."""
    # Measured at 300 dpi or more: text hinting makes low-dpi extents narrower.
    fig = plt.figure(
        figsize=(FIXED_MEASURE_SIZE, FIXED_MEASURE_SIZE),
        dpi=style.MEASURE_DPI,
    )
    grid = [_draw_row(fig, row) for row in rows]
    natural = [sum(b.left + b.width + b.right for b in row) for row in grid]
    width = width or max(
        n + 2 * style.PAGE_MARGIN + (len(row) - 1) * style.PANEL_GAP_MIN
        for n, row in zip(natural, grid, strict=True)
    )
    placements, height = _pack_rows(grid, natural, width)
    fig.set_size_inches(width, height)
    for box, x, y, _ in placements:
        box.ax.set_position(
            (
                x / width,
                1 - (y + box.height) / height,
                box.width / width,
                box.height / height,
            )
        )
    if len(placements) > 1:
        _letter_axes(fig, placements)
    return fig


def _letter_axes(fig: Figure, placements) -> None:
    """Letter each axes A, B, C, ... (only the first of a `_Joined`
    module) in reading order (by its letter's height, then left to right) at its left margin and its row's top, tagged with its panel
    module's name for `record_panel_letters`."""
    width, height = fig.get_size_inches()
    lettered = [p for p in placements if p[0].lettered]
    reading = sorted(lettered, key=lambda p: (round(p[3], 6), p[1]))
    for letter, (box, x, _, label_y) in zip(ascii_uppercase, reading, strict=False):
        fig.text(
            (x - box.left) / width,
            1 - label_y / height,
            letter,
            ha="left",
            va="top",
            fontsize=style.FONT_SIZE_PANEL_LABEL,
            fontweight="bold",
            gid=box.ax.get_label(),
        )


def _pack_rows(grid, natural, width):
    """Each axes' (box, x, y, label y): its top-left corner and its row's top
    in inches, and the figure height."""
    placements, y = [], style.PAGE_MARGIN
    for i, (row, row_width) in enumerate(zip(grid, natural, strict=True), start=1):
        top = max(style.PANEL_LABEL_SPACE, *(b.top for b in row))
        slack = width - 2 * style.PAGE_MARGIN - row_width
        gap = min(slack / max(len(row) - 1, 1), style.PANEL_GAP_MAX)
        if slack < 0 or (len(row) > 1 and gap < style.PANEL_GAP_MIN):
            print(f"Warning: row {i} has {slack / style.MM:.1f} mm to spare")
        x = (width - row_width - gap * (len(row) - 1)) / 2  # centered
        for box in row:
            if isinstance(box, _Stack | _Joined):
                placements += box.placements(x + box.left, y + top, y)
            else:
                placements.append((box, x + box.left, y + top, y))
            x += box.left + box.width + box.right + gap
        y += top + max(b.height + b.bottom for b in row) + style.ROW_GAP
    return placements, y - style.ROW_GAP + style.PAGE_MARGIN


def standalone_fixed(module_name: str) -> Figure:
    """One panel module drawn alone at its fixed axes sizes."""
    return compose_fixed([[importlib.import_module(module_name)]])


# in; a scratch canvas with room on every side of an axes for its margins
FIXED_MEASURE_SIZE = 12


@dataclass
class _AxesBox:
    """An axes, its fixed size, and the margins its decorations take (in)."""

    ax: plt.Axes
    width: float
    height: float
    left: float = 0
    right: float = 0
    top: float = 0
    bottom: float = 0
    lettered: bool = True


class _Stack:
    """Axes drawn one above another in one slot of a row, left-aligned and
    `style.ROW_GAP` apart beyond their margins (and at least
    `style.PANEL_LABEL_SPACE` for each one's letter); it packs as a single
    box."""

    def __init__(self, boxes: list[_AxesBox]):
        self.boxes = boxes
        # Each axes' top below the first's
        self.offsets = np.cumsum(
            [0]
            + [
                a.height
                + a.bottom
                + style.ROW_GAP
                + max(b.top, style.PANEL_LABEL_SPACE)
                for a, b in itertools.pairwise(boxes)
            ]
        )
        self.width = max(b.width for b in boxes)
        self.left = max(b.left for b in boxes)
        self.right = max(b.width + b.right for b in boxes) - self.width
        self.top, self.bottom = boxes[0].top, boxes[-1].bottom
        self.height = self.offsets[-1] + boxes[-1].height

    def placements(self, x: float, y: float, row_top: float) -> list[tuple]:
        """(box, x, y, label y) of each axes, the first lettered at the row's
        top and the rest at their own."""
        return [
            (
                b,
                x,
                y + dy,
                row_top if i == 0 else y + dy - max(b.top, style.PANEL_LABEL_SPACE),
            )
            for i, (b, dy) in enumerate(zip(self.boxes, self.offsets, strict=True))
        ]


class _Joined:
    """A module's axes side by side, `style.JOINED_GAP` apart whatever their
    inner margins (so the inner ones must have no y labels) and lettered once:
    one panel split into several axes. Declared by the module's `JOINED`; it
    packs as a single box."""

    def __init__(self, boxes: list[_AxesBox], lettered: bool = False):
        self.boxes = boxes
        for b in boxes[1:]:
            b.lettered = lettered
        self.offsets = np.cumsum([0] + [b.width + style.JOINED_GAP for b in boxes[:-1]])
        self.width = self.offsets[-1] + boxes[-1].width
        self.height = max(b.height for b in boxes)
        self.left, self.right = boxes[0].left, boxes[-1].right
        self.top = max(b.top for b in boxes)
        self.bottom = max(b.bottom for b in boxes)

    def placements(self, x: float, y: float, row_top: float) -> list[tuple]:
        return [
            (b, x + dx, y, row_top)
            for b, dx in zip(self.boxes, self.offsets, strict=True)
        ]


def _draw_row(fig: Figure, row: list) -> list:
    """The boxes of a row's modules, each list of modules one `_Stack`, each
    `JOINED` module one `_Joined`, each `JOIN_PREVIOUS` module one `_Joined`
    with the axes before it (each lettered), and each `STACKED` module's axes
    one `_Stack` lettered once."""
    boxes = []
    for item in row:
        if isinstance(item, list):
            boxes.append(_Stack([b for m in item for b in _draw_fixed(fig, m)]))
        elif getattr(item, "JOINED", False):
            boxes.append(_Joined(_draw_fixed(fig, item)))
        elif getattr(item, "JOIN_PREVIOUS", False) and boxes:
            previous = boxes.pop()
            assert isinstance(previous, _AxesBox)  # joins a single axes
            boxes.append(_Joined([previous, *_draw_fixed(fig, item)], lettered=True))
        elif getattr(item, "STACKED", False):
            stacked = _draw_fixed(fig, item)
            for box in stacked[1:]:
                box.lettered = False
            boxes.append(_Stack(stacked))
        else:
            boxes += _draw_fixed(fig, item)
    return boxes


def panel_axes(fig: Figure, panel: str) -> plt.Axes:
    """The axes `compose_fixed` drew panel module `panel` (its name) into,
    for adjustments spanning panels once the figure is laid out."""
    return next(ax for ax in fig.axes if ax.get_label() == panel)


def nudge_panel(
    fig: Figure,
    panel: str,
    right: float = 0.0,
    up: float = 0.0,
    grow: float = 0.0,
    letter: bool = False,
) -> None:
    """Move panel module `panel`'s axes `right` and `up` (in) once the figure
    is laid out, then raise its top `grow` (in; its bottom stays), so a panel
    can sit or reach beyond the box its row reserved. Its insets (colorbars)
    follow; its letter stays where `compose_fixed` put it unless `letter`."""
    ax = panel_axes(fig, panel)
    w, h = fig.get_size_inches()
    p = ax.get_position()
    ax.set_position((p.x0 + right / w, p.y0 + up / h, p.width, p.height + grow / h))
    if letter:
        for text in fig.texts:
            if text.get_gid() == panel:
                x, y = text.get_position()
                text.set_position((x + right / w, y + up / h))


def close_row_gap(
    fig: Figure, upper: list[str], lower: list[str], gap: float = style.PANEL_GAP_MIN
) -> None:
    """Close the empty band under the row of panel modules `upper`: everything
    above the row of panels `lower` (their letters included) moves down until
    they come within `gap` (in) of each other (`panel_room`; if not already),
    and the figure shrinks by as much."""
    room, top = panel_room(fig, upper, lower)
    open_figure_gap(fig, top + 1e-3, -max(room - gap, 0))


def panel_room(
    fig: Figure, upper: list[str], lower: list[str], lower_letters: bool = True
) -> tuple[float, float]:
    """The vertical room (in) between panel modules `upper` and the panels
    `lower` under them, from any mark (`_mark_extents`, and the panels'
    letters, unless not `lower_letters`) of one to a mark of the other it
    overlaps across the page; and the top (in, from the bottom) of `lower`'s
    marks."""
    fig.draw_without_rendering()  # places the insets
    renderer = fig.canvas.get_renderer()  # ty: ignore[unresolved-attribute]

    def extents(panels, letters=True):
        axes = [ax for ax in fig.axes if ax.get_label() in panels]
        return [b for ax in axes for b in _mark_extents(ax, renderer)] + [
            t.get_window_extent(renderer)
            for t in fig.texts
            if letters and t.get_gid() in panels
        ]

    above, below = extents(upper), extents(lower, lower_letters)
    room = min(a.y0 - b.y1 for a in above for b in below if a.x0 < b.x1 and b.x0 < a.x1)
    return room / fig.dpi, max(b.y1 for b in below) / fig.dpi


# Bars grouping whole panels of a composed figure, in a cohort's or method's
# color with its name, made room for by growing the figure at an edge
# (`grow_figure`) or between rows (`open_figure_gap`), every axes, figure text
# and figure line kept in place on the page
GROUP_BAR_WIDTH = 2.5  # pt
GROUP_NAME_GAP = 1 * style.MM  # in, between a bar and its name


def _move_on_page(fig: Figure, size, move: Callable) -> None:
    """Resize `fig` to `size` (in), each axes (but insets, which follow their
    parent), figure text and figure line moved by `move(x, y)` (in, from the
    bottom left)."""
    w, h = fig.get_size_inches()
    nw, nh = size
    for ax in fig.axes:
        if ax.get_axes_locator() is not None:
            continue
        p = ax.get_position()
        x0, y0 = move(p.x0 * w, p.y0 * h)
        ax.set_position((x0 / nw, y0 / nh, p.width * w / nw, p.height * h / nh))
    for text in fig.texts:
        x, y = move(*(np.array(text.get_position()) * (w, h)))
        text.set_position((x / nw, y / nh))
    for line in fig.lines:
        moved = [move(x * w, y * h) for x, y in np.asarray(line.get_xydata())]
        line.set_data([x / nw for x, _ in moved], [y / nh for _, y in moved])
    fig.set_size_inches(nw, nh)


def grow_figure(
    fig: Figure,
    left: float = 0.0,
    bottom: float = 0.0,
    right: float = 0.0,
    top: float = 0.0,
) -> None:
    """Grow `fig` by `left` / `bottom` / `right` / `top` inches at that edge,
    keeping everything where it was on the page."""
    w, h = fig.get_size_inches()
    _move_on_page(
        fig, (w + left + right, h + bottom + top), lambda x, y: (x + left, y + bottom)
    )


def open_figure_gap(fig: Figure, at: float, height: float) -> None:
    """Grow `fig` by `height` inches, opening a gap at `at` inches from the
    bottom: everything above `at` moves up, everything below stays (a
    negative `height` closes one)."""
    w, h = fig.get_size_inches()
    _move_on_page(fig, (w, h + height), lambda x, y: (x, y + height * (y > at)))


def _mark_extents(ax: plt.Axes, renderer) -> list:
    """The display extents of `ax`'s box, each axis (its tick labels and
    label), title, texts and legends, and its insets, one by one: finer than
    its tight bbox, which spans the empty corners between them."""
    axes = [axis.get_tightbbox(renderer) for axis in (ax.xaxis, ax.yaxis)]
    insets = [b for child in ax.child_axes for b in _mark_extents(child, renderer)]  # ty: ignore[invalid-argument-type]
    return [
        ax.bbox,
        *(m.get_window_extent(renderer) for m in _free_marks(ax)),
        *(b for b in axes if b is not None),
        *insets,
    ]


def _free_marks(ax: plt.Axes) -> list:
    """`ax`'s title and texts that have text, and its legends."""
    texts = [t for t in (ax.title, *ax.texts) if t.get_text()]
    return texts + [a for a in (*ax.artists, ax.get_legend()) if isinstance(a, Legend)]


def lowest_label(fig: Figure, axes: list) -> float:
    """The lowest point (in, from the bottom) of `axes`' ticks and labels."""
    renderer = fig.canvas.get_renderer()  # ty: ignore[unresolved-attribute]
    return min(ax.get_tightbbox(renderer).y0 for ax in axes) / fig.dpi


def highest_label(fig: Figure, axes: list) -> float:
    """The highest point (in, from the bottom) of `axes`' ticks and labels and
    of their panel letters (`_letter_axes`)."""
    renderer = fig.canvas.get_renderer()  # ty: ignore[unresolved-attribute]
    names = {ax.get_label() for ax in axes}
    boxes = [ax.get_tightbbox(renderer) for ax in axes] + [
        t.get_window_extent(renderer) for t in fig.texts if t.get_gid() in names
    ]
    return max(b.y1 for b in boxes) / fig.dpi


def _group_bar(fig: Figure, axes: list, y: float, name: str, color, above: bool):
    """A bar `y` inches up across `axes`' width, `name` over or under it."""
    h = fig.get_size_inches()[1]
    x0 = min(ax.get_position().x0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    fig.add_artist(
        Line2D(
            [x0, x1],
            [y / h] * 2,
            color=color,
            lw=GROUP_BAR_WIDTH,
            solid_capstyle="butt",
        )
    )
    fig.text(
        (x0 + x1) / 2,
        (y + (GROUP_NAME_GAP if above else -GROUP_NAME_GAP)) / h,
        name,
        ha="center",
        va="bottom" if above else "top",
        fontsize=style.FONT_SIZE_DENSE,
    )


def bar_under(fig: Figure, axes: list, y: float, name: str, color) -> None:
    """A group bar `y` inches from the bottom across `axes`, `name` under it."""
    _group_bar(fig, axes, y, name, color, above=False)


def bar_over(fig: Figure, axes: list, gap: float, name: str, color) -> None:
    """A group bar `gap` inches above `axes`' top across them, `name` over it."""
    h = fig.get_size_inches()[1]
    y = max(ax.get_position().y1 for ax in axes) * h + gap
    _group_bar(fig, axes, y, name, color, above=True)


def match_row_extent(
    fig, panel: str, first: str, last: str, axes_only: bool = False
) -> None:
    """Resize panel module `panel`'s axes so it, with its labels, spans from
    panel `first`'s left extent to panel `last`'s right, letter included; with
    `axes_only`, its axes alone span from `first`'s axes to `last`'s, its
    labels and letter left of them."""
    extent = (lambda a: a.get_window_extent()) if axes_only else _extent
    left = extent(panel_axes(fig, first)).x0
    right = extent([ax for ax in fig.axes if ax.get_label() == last][-1]).x1
    ax = panel_axes(fig, panel)
    tight, box = _extent(ax), ax.get_window_extent()
    x0 = left if axes_only else left + box.x0 - tight.x0
    x1 = right if axes_only else right - (tight.x1 - box.x1)
    position = ax.get_position()
    width = fig.bbox.width
    ax.set_position((x0 / width, position.y0, (x1 - x0) / width, position.height))
    letters = {text.get_gid(): text for text in fig.texts}
    letter_x = (x0 - (box.x0 - tight.x0)) / width
    letters[panel].set_x(letter_x if axes_only else letters[first].get_position()[0])


def _extent(ax):
    """`ax`'s extent with its labels and legends, in display pixels."""
    tight = ax.get_tightbbox()
    assert tight is not None
    return tight


def _draw_fixed(fig: Figure, module) -> list[_AxesBox]:
    """Draw `module` into its fixed-size axes and measure their margins."""
    boxes = []
    for kind in module.AXES:
        w, h = style.AXIS_SIZE[kind]
        ax = fig.add_axes((0.4, 0.4, w / FIXED_MEASURE_SIZE, h / FIXED_MEASURE_SIZE))
        ax.set_label(module.__name__.rsplit(".", 1)[-1])  # found by `panel_axes`
        boxes.append(_AxesBox(ax, w, h))
    _SOURCE["panel"] = module.__name__.rsplit(".", 1)[-1]
    module.plot([b.ax for b in boxes])
    for b in boxes:
        tight, bbox = b.ax.get_tightbbox(), b.ax.bbox
        assert tight is not None
        b.left, b.right, b.top, b.bottom = (
            max(d, 0) / fig.dpi
            for d in (
                bbox.x0 - tight.x0,
                tight.x1 - bbox.x1,
                tight.y1 - bbox.y1,
                bbox.y0 - tight.y0,
            )
        )
    return boxes


def colorbar(ax: plt.Axes, mappable, label: str, start: float = 1.0):
    """A colorbar beside `ax`, in its margin rather than taken from its width,
    `style.COLORBAR_PAD` past `start` (axes fraction; 1 is the right edge)."""
    width = ax.bbox.width / ax.figure.dpi
    cax = ax.inset_axes(
        (start + style.COLORBAR_PAD / width, 0, style.COLORBAR_WIDTH / width, 1)
    )
    return ax.figure.colorbar(mappable, cax=cax, label=label)


#: Values beyond this percentile of |value| saturate `plot_value_scatter`'s
#: color scale.
SATURATION_PERCENTILE = 99


def plot_value_scatter(
    ax, x, y, values, saturation_percentile: float = SATURATION_PERCENTILE
):
    """Cells colored by `values` on the diverging scale, saturating at
    `saturation_percentile`, the largest magnitudes drawn on top."""
    vmax = float(np.percentile(np.abs(values), saturation_percentile))
    order = np.argsort(np.abs(values))
    return ax.scatter(
        x[order],
        y[order],
        c=values[order],
        cmap=style.CMAPS["diverging"],
        vmin=-vmax,
        vmax=vmax,
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )


def legend(ax: plt.Axes, side: str = "right", **kwargs):
    """A legend outside `ax` on `side` ('right', 'left', 'above', 'below'),
    beyond its tick labels, axis labels, title, and colorbar, so like
    `colorbar` it takes margin rather than axes space. Call it once the axes
    is labeled; a plain `ax.legend()` overlays the axes instead."""
    extent = ax.get_tightbbox()
    assert extent is not None
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(extent.get_points())
    dpi = ax.figure.dpi
    px = style.LEGEND_PAD * dpi / ax.bbox.width
    py = style.LEGEND_PAD * dpi / ax.bbox.height
    # The legend corner anchored, and where, in axes units
    loc, anchor = {
        "right": ("center left", (x1 + px, 0.5)),
        "left": ("center right", (x0 - px, 0.5)),
        "above": ("lower center", (0.5, y1 + py)),
        "below": ("upper center", (0.5, y0 - py)),
    }[side]
    return ax.legend(**{"loc": loc, "bbox_to_anchor": anchor} | kwargs)


def stacked_legend(
    ax: plt.Axes, rows: list[list], side: str = "above", title=None, **kwargs
) -> list[Legend]:
    """Legends `side` of `ax` ('above' or 'below'), one centered line per
    `rows` entry (its handles) from top to bottom, the first under `title`:
    the nearest placed by `legend`, the rest stacked beyond it. `kwargs` go to
    each; returns them all."""
    above = side == "above"
    renderer = ax.figure.canvas.get_renderer()  # ty: ignore[unresolved-attribute]
    keys: list[Legend] = []
    for handles in rows[::-1] if above else rows:
        kw = {"handles": handles, "ncols": len(handles)} | kwargs
        kw["title"] = title if handles is rows[0] else None
        if not keys:
            keys.append(legend(ax, side, **kw))
            continue
        ax.add_artist(keys[-1])  # kept when the next replaces it
        extent = ax.transAxes.inverted().transform(keys[-1].get_window_extent(renderer))
        keys.append(
            ax.legend(
                loc="lower center" if above else "upper center",
                bbox_to_anchor=(0.5, extent[1, 1] if above else extent[0, 1]),
                borderaxespad=0,
                **kw,
            )
        )
    return keys


def data_border(ax: plt.Axes, y0: float, y1: float) -> None:
    """A box in the axes-line style around a diagram drawn without axes lines,
    across `ax`'s width from data height `y0` to `y1`."""
    ax.add_patch(
        Rectangle(
            (0, y0),
            1,
            y1 - y0,
            transform=blended_transform_factory(ax.transAxes, ax.transData),
            fill=False,
            edgecolor=style.BOX_COLOR,
            linewidth=style.BOX_LINE_WIDTH,
            clip_on=False,
        )
    )


def placeholder(fig: Figure | SubFigure | plt.Axes, text: str) -> None:
    """A dashed box marking a panel that is not implemented yet."""
    ax = fig if isinstance(fig, plt.Axes) else fig.subplots()
    ax.set_axis_off()
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            linestyle=style.LINE_STYLE_REFERENCE,
            edgecolor=style.COLORS["muted"],
        )
    )
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=style.COLORS["reference"],
    )


def figure_package(figure: str) -> str:
    """Dotted name of figure `figure`'s package (e.g. '2', 'S_alad_cohort')."""
    package = f"figure{figure}"
    # Supplements may be nested in their main figure's package
    (master,) = FIGURES_DIR.glob(f"**/{package}/{package}.py")
    return ".".join(("cccrise.figures", *master.parent.relative_to(FIGURES_DIR).parts))


def build_figure(figure: str, panel: str | None = None) -> Figure:
    """Figure `figure` (e.g. '2', 'S_alad_cohort'), or one of its panels."""
    style.apply_style()
    package = figure_package(figure)
    module = importlib.import_module(f"{package}.{panel or f'figure{figure}'}")
    return module.makeFigure()


def genFigure():
    """Build a figure, or one of its panels, into output/figure<N>/:

    uv run cccrise-fbuild 2                      # figures/figure2/figure2.py
    uv run cccrise-fbuild 2 rise_rank_selection  # figures/figure2/rise_rank_selection.py
    uv run cccrise-fbuild 2 --png                # PNG instead of SVG
    uv run cccrise-fbuild 2 --pdf                # a PDF beside the SVG (`svg_to_pdf`)
    uv run cccrise-fbuild S_alad_cohort        # figures/figure2/figureS_alad_cohort/
    """
    start = time.time()
    args = [arg for arg in sys.argv[1:] if arg not in ("--png", "--pdf")]
    extension = "png" if "--png" in sys.argv else "svg"
    figure, panel = args[0], (args[1] if len(args) > 1 else None)

    output_dir = OUTPUT_DIR / f"figure{figure}"
    _SOURCE["dir"] = output_dir / SOURCE_DATA_SUBDIR
    # An unlettered figure is one panel: its own module (panels override)
    _SOURCE["panel"] = panel or f"figure{figure}"
    if panel is None:  # a panel dropped from the figure leaves no data behind
        shutil.rmtree(_SOURCE["dir"], ignore_errors=True)
    fig = build_figure(figure, panel)
    output_dir.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": figures are drawn at their final print size.
    path = output_dir / f"{panel or f'figure{figure}'}.{extension}"
    _save(fig, path)
    if panel is None:
        _record_figure(figure, fig)
    print(f"Figure {' '.join(args)} is done after {time.time() - start:.1f} s.\n")


def _save(fig: Figure, path: Path) -> None:
    """`fig` saved to `path`, and with --pdf, a PDF beside an SVG."""
    fig.savefig(path)
    if "--pdf" in sys.argv and path.suffix == ".svg":
        svg_to_pdf(path)


def _record_figure(figure: str, fig: Figure) -> None:
    """A whole figure's panel letters (`record_panel_letters`) and the inputs
    its source data came from."""
    record_panel_letters(figure, fig)
    _SOURCE["dir"].mkdir(parents=True, exist_ok=True)
    (_SOURCE["dir"] / SOURCE_INPUTS).write_text(
        yaml.safe_dump(sorted(str(p) for p in _SOURCE["inputs"] | LOADED_PATHS))
    )


def svg_to_pdf(svg: Path) -> None:
    """A PDF of `svg` beside it, converted by librsvg (`rsvg-convert`) with its
    fonts embedded: matplotlib's own PDF lays out math text (R^2, p_adj) with
    other font metrics than its SVG, up to ~1 mm apart, so the PDF is drawn
    from the SVG to look the same as it (and as an Illustrator edit of it)."""
    if shutil.which("rsvg-convert") is None:
        sys.exit("--pdf needs rsvg-convert (librsvg2-bin / brew install librsvg)")
    subprocess.run(
        ["rsvg-convert", "-f", "pdf", "-o", str(svg.with_suffix(".pdf")), str(svg)],
        check=True,
    )


def record_panel_letters(figure: str, fig: Figure) -> None:
    """Set `fig<figure>` in `PANELS_PATH` to `panel_letters(fig)`."""
    panels = yaml.safe_load(PANELS_PATH.read_text()) or {}
    panels[f"fig{figure}"] = panel_letters(fig)
    PANELS_PATH.write_text(
        yaml.safe_dump(panels, sort_keys=False, default_flow_style=None)
    )


def panel_letters(fig: Figure) -> dict[str, list[str]]:
    """The letters of each panel module's axes, as tagged by `_letter_axes`
    or `label_panels`."""
    letters: dict[str, list[str]] = {}
    for text in fig.findobj(Text):
        if module := text.get_gid():
            letters.setdefault(module, []).append(text.get_text().lower())
    return {module: sorted(ls) for module, ls in letters.items()}


# Each panel's EMBO source data, saved by `source_data` only while
# `cccrise-fbuild` builds into `dir`: the panel module being drawn, the panels
# whose old files were cleared this build, and the runner outputs read
# (factors files read are in `factors.LOADED_PATHS`)
SOURCE_DATA_SUBDIR = "source_data"
SOURCE_INPUTS = "inputs.yaml"
_SOURCE: dict[str, Any] = {
    "dir": None,
    "panel": None,
    "cleared": set(),
    "inputs": set(),
}


def source_data(
    name: str, data: pd.DataFrame | pd.Series, description: str, part: int | None = None
) -> None:
    """Save `data`, values the panel being drawn plots, as its EMBO source
    data file `<name>.csv`, explained in the panel's README by `description`
    (which should name any column whose meaning or units its name doesn't
    make plain). Give the individual values behind any summary drawn, and
    name any integer index that labels rows (an unnamed one is dropped).
    Name files `<assay>_<entity>` in snake case, e.g. `bicv_r2x_by_rise_rank`
    or `de_sender_vs_rest`. `part`
    is the index of the panel module's lettered axes the file belongs to (1:
    the C of a module lettered B-C); by default it spans them all."""
    root, panel = _SOURCE["dir"], _SOURCE["panel"]
    if root is None:
        return
    folder = root / panel
    if panel not in _SOURCE["cleared"]:
        shutil.rmtree(folder, ignore_errors=True)
        _SOURCE["cleared"].add(panel)
    folder.mkdir(parents=True, exist_ok=True)
    frame = data.to_frame() if isinstance(data, pd.Series) else data
    # Index labels kept as columns, but not an unnamed integer index: row
    # positions left over from filtering or sorting
    positional = frame.index.names == [None] and frame.index.dtype.kind in "iu"
    frame = frame.reset_index(drop=positional)
    # Value by value: `replace` fails on array cells (GroupSizes) and categoricals
    for c in frame.select_dtypes(["object", "string", "category"]):
        frame[c] = frame[c].map(
            lambda v: "CCC-RISE" if isinstance(v, str) and v == "cccRISE" else v
        )
    frame.to_csv(folder / f"{name}.csv", index=False)
    (folder / f"{name}.yaml").write_text(
        yaml.safe_dump({"description": description, "part": part})
    )


# =============================================================================
# Shared panels
# =============================================================================

SIG_ALPHA = 0.05


def p_text(p: float, name: str = "p") -> str:
    """`p` for a panel: "p = 0.03", or "p = 1 × 10⁻⁵" below 0.001, one digit."""
    if p >= 1e-3:
        return f"{name} = {p:.1g}" if p < 0.01 else f"{name} = {p:.2f}"
    mantissa, exponent = f"{p:.0e}".split("e")
    return rf"{name} = ${mantissa} \times 10^{{{int(exponent)}}}$"


#: Seed of every strip plot's jitter (`stripplot`).
JITTER_SEED = 0
# Axis-label names of outcome `obs` columns
OUTCOME_LABELS = {"6monthcondition": "6-mo. outcome"}


RANK_LINE: dict[str, Any] = {
    "color": style.COLORS["reference"],
    "linestyle": style.LINE_STYLE_REFERENCE,
    "linewidth": style.LINE_WIDTH_THIN,
}


def stripplot(**kwargs):
    """`sns.stripplot`, its jitter seeded by `JITTER_SEED` so each build
    draws the same. Seaborn jitters from NumPy's global random state, which
    is restored afterwards."""
    state = np.random.get_state()
    np.random.seed(JITTER_SEED)
    try:
        return sns.stripplot(**kwargs)
    finally:
        np.random.set_state(state)


def mark_rank(ax: plt.Axes, rank: int | None, top: float = 1) -> None:
    """Vertical reference line at the rank the canonical fit uses, up to `top`
    (axes fraction) to leave room for a key above it."""
    if rank is not None:
        ax.axvline(rank, ymax=top, zorder=0, **RANK_LINE)


def reference_line(
    ax: plt.Axes,
    value: float,
    horizontal: bool = True,
    linestyle: str = style.LINE_STYLE_REFERENCE,
    label: str | None = None,
):
    """Null, chance, or threshold line in the shared reference style."""
    return (ax.axhline if horizontal else ax.axvline)(
        value,
        color=style.COLORS["reference"],
        linestyle=linestyle,
        linewidth=style.LINE_WIDTH_THIN,
        label=label,
    )


# --- Rank selection (ALAD Fig 2, IBD Fig 4) ------------------------------------


def plot_rise_rank(sweep: pd.DataFrame, ax: plt.Axes, rank: int | None = None):
    """In-sample R2X of the full-data fit, and in-sample (training block) and
    held-out BiCV R2X as mean +/- SD over the BiCV repeats, against RISE
    rank, from `cccrise-riserank`. BiCV peaks or plateaus near the rank that
    generalizes to unseen cells and genes. Returns the three mean lines."""
    full = sweep[sweep["Metric"] == "Fit R2X"].groupby("Rank")["R2X"].mean()
    lines = ax.plot(full.index, full, color=style.COLORS["fit_full"], label="Fit")
    repeats = sweep[sweep["Metric"] == "BiCV R2X"].groupby("Rank")
    for column, label, color in (
        ("Train Block R2X", "BiCV train", style.COLORS["fit"]),
        ("R2X", "BiCV test", style.COLORS["heldout"]),
    ):
        mean, sd = repeats[column].mean(), repeats[column].std().fillna(0)
        lines.append(plot_mean_sd(ax, mean.index, mean, sd, color, label))
    mark_rank(ax, rank)
    ax.set_xlabel("RISE rank")
    ax.set_ylabel("R²X")
    ax.legend(handles=lines)
    return lines


def plot_mean_sd(ax: plt.Axes, x, mean, sd, color: str, label: str, **kwargs):
    """`mean` against `x` with a +/- `sd` `plot_band`; returns the mean line."""
    return plot_band(ax, x, mean, mean - sd, mean + sd, color, label, **kwargs)


def plot_band(ax: plt.Axes, x, center, lower, upper, color: str, label: str, **kwargs):
    """`center` against `x` with a shaded, thinly outlined band from `lower`
    to `upper`; `kwargs` go to the center line, which is returned."""
    (line,) = ax.plot(x, center, color=color, label=label, **kwargs)
    for bound in (lower, upper):
        ax.plot(x, bound, color=color, linewidth=style.LINE_WIDTH_THIN)
        ax.fill_between(x, center, bound, color=color, alpha=0.25, lw=0)
    return line


def plot_cpd_fit_r2x(
    sweep: pd.DataFrame,
    ax: plt.Axes,
    rank: int | None = None,
    color: str = style.COLORS["fit"],
    marker: str | None = "o",
):
    """Full-tensor CP fit R2X against CPD rank, from `cccrise-cpdrank`."""
    ax.plot(sweep["Rank"], sweep["FitR2X"], marker=marker, color=color, label="Fit")
    mark_rank(ax, rank)
    ax.set_xlabel("CPD rank")
    ax.set_ylabel("Fit R²X")


def plot_cpd_imputation_r2x(repeats: pd.DataFrame, ax: plt.Axes):
    """Overlay k-fold imputation R2X (mean +/- SD over fold splits) on a
    `plot_cpd_fit_r2x` axis, from `load_cpd_imputation`."""
    by_rank = repeats.groupby("Rank")["ImputationR2X"]
    mean, sd = by_rank.mean(), by_rank.std().fillna(0)
    plot_mean_sd(ax, mean.index, mean, sd, style.COLORS["heldout"], "Imputation")
    ax.set_ylabel("R²X")
    ax.set_ylim(0, 1.05 * ax.dataLim.y1)
    ax.legend(loc="lower right", borderaxespad=0.1)


# --- CPD rank selection across seeded starts (ALAD Fig 2, IBD Fig 4) -----------
# Drawn from `cccrise-cpdrank`'s seeded starts: fit R2X across starts, each
# mode's collinearity, and the distinct minima the starts reach.

CP_MODES = {"A": "Condition", "B": "Sender", "C": "Receiver", "D": "LR pair"}
KNEEDLE_EDGE = "#e34948"
#: Whether a start reaches its rank's most common solution -> color, label
SOLUTION_COLORS = {True: style.CATEGORICAL[0], False: style.COLORS["muted"]}
SOLUTION_LABELS = {True: "Most common solution", False: "Other solutions"}
COLLINEARITY_BOX_COLOR = "0.9"
#: The selected rank's marker above its bar in the minima panel
RANK_MARKER: dict[str, Any] = {
    "marker": "v",
    "color": style.COLORS["reference"],
    "markersize": 1.3 * style.MARKER_SIZE,
    "linestyle": "",
}


def load_minima(dataset: str) -> pd.DataFrame:
    """`rank_selection.distinct_minima` of `dataset`'s starts."""
    start_fms = load_runner_output(
        DATASETS[dataset].cpd_start_fms_filename(), RANK_SELECTION_DIR
    )
    r2x = load_cpd_rank(dataset).set_index(["Rank", "Seed"])["FitR2X"]
    return distinct_minima(start_fms, r2x)


def rank_axis(ax: plt.Axes, ranks) -> None:
    ranks = np.asarray(ranks)
    ax.set_xticks(ranks[ranks % 2 == 1] if ranks.max() > 10 else ranks)
    ax.set_xlim(0.4, ranks.max() + 0.6)
    ax.set_xlabel("CPD rank")


def plot_r2x_starts(ax: plt.Axes, dataset: str, rank: int | None = None) -> None:
    """`plot_r2x_sweep` of `dataset`'s CP sweep."""
    plot_r2x_sweep(ax, load_cpd_rank(dataset), rank)


def plot_r2x_sweep(ax: plt.Axes, sweep: pd.DataFrame, rank: int | None = None) -> None:
    """Fit R2X against CPD rank of a seeded-start `sweep` ('Rank', 'Start',
    'Seed', 'FitR2X'): the median over starts, banded by their range."""
    source_data(
        "cpd_fit_r2x_by_rank_and_start",
        sweep[["Rank", "Start", "Seed", "FitR2X"]],
        "Fit R2X of every seeded SVD start at every CPD rank (start 0 is the "
        "production fit); plotted as the median over starts, banded by their "
        "range.",
    )
    by_rank = sweep.groupby("Rank")["FitR2X"]
    median = by_rank.median()
    color = style.COLORS["fit"]
    ax.fill_between(
        median.index, by_rank.min(), by_rank.max(), color=color, alpha=0.2, lw=0
    )
    ax.plot(median.index, median, color=color, label="Median start")
    mark_rank(ax, rank)
    rank_axis(ax, median.index)
    ax.set_ylabel("Fit R²X")


def plot_collinearity_boxes(
    ax: plt.Axes,
    dataset: str,
    rank: int | None = None,
    part: int | None = None,
    wrap_key: bool = False,
    key_loc: LegendLocType | None = "lower right",
) -> None:
    """Per CPD rank, the largest |cosine| between two components' LR pair
    factors (`plot_start_boxes`). `part` is `source_data`'s."""
    sweep = _collinearity_sweep(dataset, ["D"], part)
    minima = load_minima(dataset)
    plot_start_boxes(ax, sweep, "MaxCosD", minima, rank, wrap_key, key_loc)
    ax.set_ylabel("LR pair max |cosine|")


def plot_start_boxes(
    ax: plt.Axes,
    sweep: pd.DataFrame,
    column: str,
    minima: pd.DataFrame,
    rank: int | None = None,
    wrap_key: bool = False,
    key_loc: LegendLocType | None = "lower right",
) -> None:
    """Per CPD rank, `column` of a seeded-start `sweep` ('Rank', 'Seed'): a
    box over the starts with each start overlaid, colored by whether it
    reaches its rank's most common solution (`minima`, as
    `rank_selection.distinct_minima`) and keyed at `key_loc` (no key if
    None), its labels one word per line if `wrap_key`."""
    common_seeds = minima.query("Order == 1").set_index("Rank")["Seeds"]
    sweep = sweep.assign(
        Common=[s in common_seeds[r] for r, s in zip(sweep["Rank"], sweep["Seed"])]
    ).sort_values("Common", ascending=False, kind="stable")  # other solutions on top
    kw = {"data": sweep, "x": "Rank", "y": column, "native_scale": True, "ax": ax}
    sns.boxplot(
        **kw,
        color=COLLINEARITY_BOX_COLOR,
        width=0.7,
        showfliers=False,
        linewidth=style.LINE_WIDTH_THIN,
    )
    stripplot(
        **kw,
        hue="Common",
        hue_order=[True, False],
        palette=SOLUTION_COLORS,
        legend=False,
        jitter=0.2,
        size=1.8,
        edgecolor=style.COLORS["ink"],
        linewidth=style.LINE_WIDTH_MIN,
    )
    _collinearity_axes(ax, sweep, rank)
    if key_loc is None:
        return
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            markersize=style.MARKER_SIZE,
            color=SOLUTION_COLORS[c],
            label=SOLUTION_LABELS[c].replace(" ", "\n" if wrap_key else " "),
        )
        for c in (True, False)
    ]
    ax.legend(
        handles=handles,
        loc=key_loc,
        fontsize=style.FONT_SIZE_DENSE,
        handlelength=0.5,
        handletextpad=0.3,
        borderaxespad=0.1,
    )


def plot_collinearity_medians(
    ax: plt.Axes,
    dataset: str,
    modes: Sequence[str] = tuple(CP_MODES),
    rank: int | None = None,
    part: int | None = None,
    key_rank: bool = True,
) -> None:
    """Per CPD rank and mode in `modes` (`CP_MODES` keys), the largest |cosine|
    between two components: the median over the seeded starts in an
    interquartile-range band, keyed at the lower right with `rank`'s line if
    `key_rank`. `part` is `source_data`'s."""
    sweep = _collinearity_sweep(dataset, modes, part)
    handles = []
    for mode in modes:
        by_rank = sweep.groupby("Rank")[f"MaxCos{mode}"]
        median = by_rank.median()
        line = plot_band(
            ax,
            median.index,
            median,
            by_rank.quantile(0.25),
            by_rank.quantile(0.75),
            style.CATEGORICAL[list(CP_MODES).index(mode)],
            CP_MODES[mode],
            marker="o",
            markersize=style.MARKER_SIZE,
        )
        handles.append(line)
    _collinearity_axes(ax, sweep, rank)
    ax.set_ylabel("Max |cosine|")
    if rank is not None and key_rank:
        handles.append(Line2D([], [], label="Selected rank", **RANK_LINE))
    ax.legend(handles=handles, loc="lower right", fontsize=style.FONT_SIZE_DENSE)


def _collinearity_sweep(
    dataset: str, modes: Sequence[str], part: int | None
) -> pd.DataFrame:
    """`dataset`'s CP sweep from rank 2 (one component has no pair), its
    `modes`' collinearity saved as source data: the LR pair mode's apart, as
    panels drawing only it share it with those drawing every mode."""
    sweep = load_cpd_rank(dataset)
    keys = ["Rank", "Start", "Seed"]
    if "D" in modes:
        source_data(
            "cpd_lr_pair_max_cosine_by_rank",
            sweep[[*keys, "MaxCosD"]],
            "Per CPD rank and seeded SVD start, the largest absolute cosine "
            "between two components' LR pair loadings.",
            part=part,
        )
    if others := [f"MaxCos{m}" for m in modes if m != "D"]:
        source_data(
            "cpd_max_cosine_by_rank",
            sweep[[*keys, *others]],
            "Per CPD rank and seeded SVD start, the largest absolute cosine "
            "between two components in each other mode (MaxCosA condition, B "
            "sender, C receiver).",
            part=part,
        )
    return sweep[sweep["Rank"] > 1]


def _collinearity_axes(ax: plt.Axes, sweep: pd.DataFrame, rank: int | None) -> None:
    mark_rank(ax, rank)
    rank_axis(ax, np.arange(1, sweep["Rank"].max() + 1))
    ax.set_ylim(0, 1.05)


def plot_minima(
    ax: plt.Axes, dataset: str, rank: int | None = None, part: int | None = None
) -> None:
    """`plot_minima_table` of `dataset`'s starts."""
    plot_minima_table(ax, load_minima(dataset), rank, part)


def plot_minima_table(
    ax: plt.Axes, table: pd.DataFrame, rank: int | None = None, part: int | None = None
) -> None:
    """The starts reaching each distinct minimum at each CPD rank (`table`, as
    `rank_selection.distinct_minima`), stacked from the one most starts
    reach (blue; of two tied, the one with the higher R2X) up, the others
    grey (keyed by `plot_start_boxes`), with `rank` marked above its bar."""
    source_data(
        "cpd_minima_by_rank",
        table,
        "Distinct minima per CPD rank (starts joined at FMS >= 0.95): the "
        "starts reaching each, its best R2X, its FMS to the minimum most "
        "starts reach (where two minima tie for most, 'Tied', to the one with "
        "the higher R2X), its order by starts (1 = most; ties: higher R2X), "
        "and the seeds of the starts reaching it.",
        part=part,
    )
    for _, g in table.groupby("Rank"):
        ax.bar(
            g["Rank"],
            g["Starts"],
            bottom=g["Starts"].cumsum() - g["Starts"],
            width=0.72,
            color=[SOLUTION_COLORS[o == 1] for o in g["Order"]],
            edgecolor="white",
            lw=0.6,
        )
    n_starts = int(table.groupby("Rank")["Starts"].sum().max())
    if rank is not None:
        ax.plot(rank, 1.07 * n_starts, clip_on=False, **RANK_MARKER)
    ax.set_ylim(0, n_starts)
    ax.set_yticks(np.arange(0, n_starts + 1, n_starts // 2))
    ax.set_ylabel("Number of starts\nreaching each solution")
    rank_axis(ax, table["Rank"].unique())


# --- Outcome associations (ALAD Fig 3, IBD Fig 4) ------------------------------

#: Outcome levels in reading order (healthy/control first); unlisted levels
#: follow alphabetically.
LEVEL_ORDER = [
    "Healthy",
    "CD_uninflamed",
    "CD_inflamed",
    "control",
    "declined",
    "recovered",
    "No CLAD/Stable",
    "No CLAD/Recovered",
    "Stable/Recovered",
    "No CLAD",
    "CLAD",
    "CLAD/Declined",
    "CLAD/Stable",
    "CLAD/Died",
    "Alive",
    "Died",
    "no",
    "yes",
    "stable",
    "improved",
    "ILD",
    "Other",
    "IgG-",
    "IgG+",
    "female",
    "male",
]


def plot_association_heatmap(
    table: pd.DataFrame,
    ax: plt.Axes,
    xlabel: str = "CPD component",
    p_column: str = "PAdj",
    p_label: str = "FDR",
    annotate: bool = True,
) -> None:
    """Eta-squared of each component (columns) against each outcome
    association (rows), both in `table` order, from
    `anova.run_multiple_factors_anova`; bold with a star where `p_column` is
    below `SIG_ALPHA`. Without `annotate` (too many columns for the values)
    only the stars are written."""
    rows, columns = (
        list(dict.fromkeys(table[c])) for c in ("Association", "Component")
    )
    effect, padj = (
        table.pivot(index="Association", columns="Component", values=v).reindex(
            index=rows, columns=columns
        )
        for v in ("EtaSquared", p_column)
    )
    values = effect.to_numpy(dtype=float)
    vmax = np.nanmax(values)
    image = ax.imshow(
        values, cmap=style.CMAPS["weight"], vmin=0, vmax=vmax, aspect="auto"
    )
    _label_association_cells(ax, values, padj.to_numpy() < SIG_ALPHA, annotate)
    ax.set_xticks(range(effect.shape[1]), effect.columns)
    ax.set_yticks(range(len(rows)), rows, fontsize=style.FONT_SIZE_DENSE)
    ax.set_xlabel(xlabel)
    colorbar(ax, image, rf"$\eta^2$ (* {p_label} < {SIG_ALPHA})")


def _label_association_cells(
    ax: plt.Axes, values: np.ndarray, significant: np.ndarray, annotate: bool
) -> None:
    """Each finite cell's value (if `annotate`), starred and bold where
    `significant`."""
    vmax = np.nanmax(values)
    for (i, j), value in np.ndenumerate(values):
        if not np.isfinite(value) or not (annotate or significant[i, j]):
            continue
        ax.text(
            j,
            i,
            (f"{value:.2f}" if annotate else "") + ("*" if significant[i, j] else ""),
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold" if significant[i, j] else "normal",
            color="white" if value > 0.6 * vmax else style.COLORS["ink"],
        )


def plot_association_effect_sizes(
    table: pd.DataFrame,
    ax: plt.Axes,
    highlight_component: int | None = None,
    emphasis_color: str = style.COLORS["emphasis"],
    muted_color: str = style.COLORS["muted"],
) -> None:
    """Eta-squared per CP component for one association, starred where the
    adjusted p is below `SIG_ALPHA`. Significant bars are emphasized, or only
    `highlight_component` when given."""
    rows = table.sort_values("Component")
    components = rows["Component"].astype(int).to_numpy()
    p_values = rows["PAdj" if "PAdj" in rows else "P"].to_numpy(dtype=float)
    significant = p_values < SIG_ALPHA
    emphasized = (
        significant
        if highlight_component is None
        else components == highlight_component
    )
    values = rows["EtaSquared"].to_numpy(dtype=float)
    x = np.arange(len(components))
    ax.bar(x, values, color=np.where(emphasized, emphasis_color, muted_color))
    for xi, value in zip(x[significant], values[significant], strict=True):
        ax.text(xi, value, "*", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x, components)
    ax.set_xlabel("CPD component")
    ax.set_ylabel(r"$\eta^2$")
    ax.set_ylim(0, np.nanmax(values) * 1.2)
    if "Association" in rows:
        ax.set_title(str(rows["Association"].iloc[0]))


def group_loadings(
    adata: anndata.AnnData,
    component: int,
    group_column: str,
    condition_column: str,
    read_depth_correct: bool = True,
    **regrouping,
) -> pd.DataFrame:
    """Loading (`uns["A"]`, read-depth-corrected unless `read_depth_correct`
    is False) of one component per condition, with its (regrouped, see
    `anova.prepare_group_labels`) outcome level; conditions without a level
    are dropped."""
    labels = prepare_group_labels(
        get_sample_to_group(adata, condition_column, group_column), **regrouping
    )
    loadings = correct_conditions(adata) if read_depth_correct else adata.uns["A"]
    return pd.DataFrame(
        {
            "Condition": labels.index,
            "Status": labels.to_numpy(),
            "Loading": loadings[:, component - 1],
        }
    ).dropna(subset=["Status"])


#: Outcome levels as figures and the app display them; the source data keep
#: the original labels.
LEVEL_LABELS = {
    "No CLAD/Stable": "Stable",
    "No CLAD/Recovered": "Recovered",
    "CD_uninflamed": "CD uninflamed",
    "CD_inflamed": "CD inflamed",
}


def level_label(level) -> str:
    return LEVEL_LABELS.get(str(level), str(level))


def ordered_levels(levels) -> list:
    """Distinct `levels` in `LEVEL_ORDER`, unlisted ones after alphabetically."""
    rank = {level: i for i, level in enumerate(LEVEL_ORDER)}
    return sorted(set(levels), key=lambda level: (rank.get(level, len(rank)), level))


def _join_patients(ax, points, adata, condition_column, patient_column) -> None:
    """Line through the points of conditions that share a patient."""
    patient = get_sample_to_group(adata, condition_column, patient_column)
    points = points.assign(Patient=patient.loc[points["Condition"]].to_numpy())
    for _, samples in points.groupby("Patient"):
        if len(samples) > 1:
            ax.plot(
                samples["x"],
                samples["Loading"],
                color=style.COLORS["reference"],
                linewidth=style.LINE_WIDTH_THIN,
                zorder=2,
            )


def plot_category_component_weights(
    adata: anndata.AnnData,
    component: int,
    group_column: str,
    ax: plt.Axes,
    xlabel: str | None = None,
    anova_df: pd.DataFrame | None = None,
    patient_column: str | None = None,
    condition_column: str = "dsco_id",
    palette: Sequence[str] = style.CATEGORICAL,
    read_depth_correct: bool = True,
    show_counts: bool = True,
    **regrouping,
) -> None:
    """Box plot of one component's condition loadings per outcome level,
    with the conditions as jittered points, joined per patient when
    `patient_column` is given. `regrouping` (mapping / levels /
    control_label / other_label) is passed to `anova.prepare_group_labels`;
    `anova_df` adds the component's F, adjusted p, and eta-squared in the
    corner; `xlabel` (default `group_column`, "" for none) names the outcome;
    `show_counts` adds each level's sample count to its tick label."""
    points = group_loadings(
        adata,
        component,
        group_column,
        condition_column,
        read_depth_correct,
        **regrouping,
    )
    levels = ordered_levels(points["Status"])
    counts = points["Status"].value_counts()
    sns.boxplot(
        data=points,
        x="Status",
        y="Loading",
        order=levels,
        hue="Status",
        hue_order=levels,
        palette=list(palette[: len(levels)]),
        legend=False,
        width=0.5,
        fliersize=0,
        linewidth=style.LINE_WIDTH_THIN,
        boxprops={"alpha": 0.8},
        ax=ax,
    )
    # Jittered by hand so patient lines can run through the exact points.
    rng = np.random.default_rng(0)
    points["x"] = points["Status"].map({level: i for i, level in enumerate(levels)})
    points["x"] += rng.uniform(-0.2, 0.2, len(points))
    ax.scatter(
        points["x"],
        points["Loading"],
        color=style.COLORS["ink"],
        s=style.POINT_SIZE,
        zorder=3,
    )
    if patient_column is not None:
        _join_patients(ax, points, adata, condition_column, patient_column)

    # Wrapped at "/" rather than rotated, so long levels stay legible.
    labels = [level_label(level).replace("/", "/\n") for level in levels]
    if show_counts:
        labels = [
            f"{t}\n(n={counts[lv]})" for t, lv in zip(labels, levels, strict=True)
        ]
    ax.set_xticks(range(len(levels)), labels)
    ax.set_xlabel(group_column if xlabel is None else xlabel)
    ax.set_ylabel(f"Component {component} loading")
    ax.text(
        0.03,
        0.97,
        _anova_stats(anova_df, component),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style.FONT_SIZE_LABEL,
    )


def _anova_stats(anova_df: pd.DataFrame | None, component: int) -> str:
    """Title suffix with one component's F, adjusted p, and eta-squared."""
    if anova_df is None:
        return ""
    row = anova_df[anova_df["Component"] == component]
    if row.empty:
        return ""
    f, padj, eta = (float(row[c].iloc[0]) for c in ("F", "PAdj", "EtaSquared"))
    return f"$F$ = {f:.2f}\n$p_{{adj}}$ = {padj:.2g}\n$\\eta^2$ = {eta:.2f}"


# =============================================================================
# Cell type granularity: eigenstate and per-cell sender/receiver weights
# =============================================================================

STRONG_CELL_PERCENTILE = 90  # Shiny explorer's default Strength Threshold


@functools.cache
def get_cell_projections(dataset: str) -> anndata.AnnData:
    """`dataset`'s factors with per-cell eigenstate projections
    (`obsm["projections"]`) and sender/receiver weights (`obsm["sc_B"]`,
    `obsm["rc_C"]`), cached so the panels of one figure project once."""
    spec = DATASETS[dataset]
    return load_factors_to_anndata(spec.factors_path(), adata=spec.load())


def plot_component_eigenstate_loadings(
    adata: anndata.AnnData, component: int, ax: plt.Axes, labeled: Sequence[int]
) -> None:
    """A CP component's sender (B) and receiver (C) loading on each RISE
    eigenstate, with only the `labeled` eigenstates' ticks labeled."""
    sender, receiver = (np.asarray(adata.uns[k])[:, component - 1] for k in "BC")
    eigenstates = np.arange(1, len(sender) + 1)
    width = 0.4
    for offset, loading, role in ((-1, sender, "sender"), (1, receiver, "receiver")):
        ax.bar(
            eigenstates + offset * width / 2,
            loading,
            width,
            color=style.COLORS[role],
            label=f"{role.capitalize()} ({'B' if role == 'sender' else 'C'})",
        )
    ax.axhline(0, color=style.COLORS["reference"], linewidth=style.LINE_WIDTH_THIN)
    ax.set_xticks(sorted(labeled))
    ax.set(xlabel="RISE eigen-state", ylabel="Loading")
    ax.legend(loc="upper left")


def plot_celltype_scatter(
    adata: anndata.AnnData,
    dataset: str,
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    highlight: Sequence[str] = (),
    thresholds: bool = True,
) -> None:
    """Per-cell `x` vs. `y` colored by cell type, the most abundant types
    drawn beneath, the `highlight` cell types enlarged on top, and, with
    `thresholds`, dashed lines at each axis's `STRONG_CELL_PERCENTILE`."""
    celltypes = adata.obs[DATASETS[dataset].celltype_column].astype(str).to_numpy()
    colors = style.CELL_TYPE_COLORS[dataset]
    counts = pd.Series(celltypes).value_counts()
    order = np.argsort(-counts[celltypes].to_numpy(), kind="stable")
    ax.scatter(
        x[order],
        y[order],
        c=[colors[c] for c in celltypes[order]],
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    for celltype in highlight:
        is_highlight = celltypes == celltype
        ax.scatter(
            x[is_highlight],
            y[is_highlight],
            c=colors[celltype],
            s=style.POINT_SIZE_BACKGROUND,
            linewidths=0,
            rasterized=True,
        )
    if not thresholds:
        return
    for values, line in ((x, ax.axvline), (y, ax.axhline)):
        line(
            np.percentile(values, STRONG_CELL_PERCENTILE),
            color=style.COLORS["reference"],
            linestyle=style.LINE_STYLE_REFERENCE,
            linewidth=style.LINE_WIDTH_THIN,
        )


def celltype_handles(dataset: str) -> list[Patch]:
    """Legend handles of `dataset`'s cell type colors."""
    return [
        Patch(color=color, label=celltype)
        for celltype, color in style.CELL_TYPE_COLORS[dataset].items()
    ]


# =============================================================================
# Factor matrix heatmaps (CP: Figure 2; RISE: its factor supplement)
# =============================================================================

# Row labels overlaid on a factor heatmap: below FONT_SIZE_DENSE so the names
# fit inside it, stacked LABEL_GAP apart, each LEADER_LENGTH (columns) from
# its cell
FACTOR_LABEL_FONT_SIZE = 5
FACTOR_LABEL_GAP = 1.9 * style.MM
FACTOR_LEADER_LENGTH = 0.2


def plot_factor_heatmap(ax: plt.Axes, loadings: np.ndarray, xlabel: str):
    """A factor matrix (rows x components, in the given row order), each
    component divided by its largest magnitude (keeping sign), on the
    diverging scale; returns the image for a shared colorbar."""
    image = ax.imshow(
        loadings / np.abs(loadings).max(axis=0),
        cmap=style.CMAPS["diverging"],
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    ax.autoscale(False)  # overlaid labels and leaders never shrink the matrix
    components = np.arange(loadings.shape[1])
    ax.set_xticks(components, components + 1)
    ax.set_xlabel(xlabel)
    return image


#: ALAD condition groupings drawn as color strips beside condition factor
#: matrices: obs column -> (legend title, {level: color}). Status in greys so
#: its colors don't collide with the outcomes' (Fig. 3A's)
CONDITION_STRIPS = {
    "ALADstatus": (
        "ALAD status",
        {"control": "0.85", "recovered": "0.55", "declined": "0.25"},
    ),
    "6monthcondition": (
        "6-mo outcome",
        dict(
            zip(
                ("No CLAD/Stable", "No CLAD/Recovered", "CLAD", "Died"),
                style.CATEGORICAL,
                strict=False,
            )
        ),
    ),
}
STRIP_WIDTH = 1.2 * style.MM
STRIP_GAP = 0.4 * style.MM


def group_conditions(
    groups: pd.DataFrame, order: np.ndarray, strips: dict = CONDITION_STRIPS
) -> np.ndarray:
    """Conditions (rows of `groups`, their obs) in `order` (e.g.
    `dominant_order`) regrouped by each `strips` column in turn, levels in
    its order."""
    ranks = [
        groups[column].map({level: i for i, level in enumerate(colors)}).to_numpy()
        for column, (_, colors) in strips.items()
    ]
    return np.lexsort((np.argsort(order), *ranks[::-1]))


def condition_strips(
    ax: plt.Axes,
    groups: pd.DataFrame,
    strips: dict = CONDITION_STRIPS,
    brackets: bool = False,
) -> None:
    """A color strip per `strips` column left of a condition factor matrix
    (rows in `groups`' order), the y label moved left of them, and their
    legend, titled per strip, left of the y label; with `brackets`, instead
    of the legend each level is named right of the matrix at a bracket
    spanning its rows (`_strip_brackets`)."""
    inches = ax.bbox.width / ax.figure.dpi
    step = (STRIP_WIDTH + STRIP_GAP) / inches
    for i, (column, (_, colors)) in enumerate(reversed(strips.items())):
        strip = ax.inset_axes((-(i + 1) * step, 0, STRIP_WIDTH / inches, 1))
        rgb = [to_rgb(colors[level]) for level in groups[column]]
        strip.imshow(np.array(rgb)[:, None], aspect="auto", interpolation="nearest")
        strip.set(xticks=[], yticks=[])
        strip.spines[:].set_visible(False)
    strip.set_ylabel(ax.get_ylabel())  # the leftmost strip
    ax.set_ylabel("")
    if brackets:
        _strip_brackets(ax, groups, strips)
        return
    titles = [title for title, _ in strips.values()]
    entries = [
        entry
        for title, colors in strips.values()
        for entry in [
            Patch(color="none", label=title),
            *(
                Patch(color=c, label=_sentence_case(level_label(level)))
                for level, c in colors.items()
            ),
        ]
    ]
    key = legend(ax, "left", handles=entries, handlelength=0.8)
    for text in key.get_texts():
        if text.get_text() in titles:
            text.set_fontweight("bold")
    _center_titles(ax, key, titles)


def _center_titles(ax: plt.Axes, key: Legend, titles: list[str]) -> None:
    """Center `key`'s `titles` rows (their empty handles dropped), and its
    other rows as one left-aligned block, on a common axis, by padding each
    row's left."""
    renderer = ax.figure.canvas.get_renderer()  # ty: ignore[unresolved-attribute]
    rows = [
        row
        for column in key._legend_handle_box.get_children()  # ty: ignore[unresolved-attribute]
        for row in column.get_children()
    ]
    is_title = [row.get_children()[1].get_text() in titles for row in rows]
    for row in itertools.compress(rows, is_title):
        del row.get_children()[0]
    widths = np.array([row.get_bbox(renderer).width for row in rows])
    entries = ~np.array(is_title)
    # Entry rows share the block's width, so their handles stay aligned
    widths[entries] = widths[entries].max()
    pads = (widths.max() - widths) / 2 / renderer.points_to_pixels(1.0)
    for row, pad in zip(rows, pads, strict=True):
        row.get_children().insert(0, DrawingArea(pad, 0))


#: `_strip_brackets`: bracket and name right of the matrix, and a tick (in)
BRACKET_X, BRACKET_NAME_X, BRACKET_TICK = 0.6 * style.MM, 1.6 * style.MM, 0.5 * style.MM
BRACKET_NAME_ROTATION = 45  # degrees: names angled up, narrowing the margin


def _runs(levels: list) -> list[tuple[object, int, int]]:
    """(level, first row, last row) of each run of equal `levels`."""
    out, start = [], 0
    for i in range(1, len(levels) + 1):
        if i == len(levels) or levels[i] != levels[start]:
            out.append((levels[start], start, i - 1))
            start = i
    return out


def _strip_brackets(ax: plt.Axes, groups: pd.DataFrame, strips: dict) -> None:
    """Each run of a strip's level named, angled up in its color, right of the matrix at
    a bracket spanning its rows; a name that would crowd the one above moves
    down on a leader line."""
    width = ax.bbox.width / ax.figure.dpi
    height = ax.bbox.height / ax.figure.dpi
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    bx, nx, tick = (1 + v / width for v in (BRACKET_X, BRACKET_NAME_X, BRACKET_TICK))
    low, high = sorted(ax.get_ylim())
    need = 2.4 * style.MM * (high - low) / height  # one name's height, in rows
    line: dict[str, Any] = {"transform": trans, "clip_on": False}
    for column, (_, colors) in strips.items():
        last = -np.inf
        for level, first, end in _runs(list(groups[column])):
            color = colors[level]
            top, bottom = first + 0.1, end - 0.1
            ax.plot(
                [bx, bx],
                [top, bottom],
                color=color,
                lw=0.9,
                solid_capstyle="butt",
                **line,
            )
            for y in (top, bottom):
                ax.plot([bx - (tick - 1), bx], [y, y], color=color, lw=0.9, **line)
            middle = (first + end) / 2
            y = max(middle, last + need)
            last = y
            if y != middle:
                ax.plot(
                    [bx, nx - 0.3 * (nx - bx)], [middle, y], color=color, lw=0.5, **line
                )
            ax.text(
                nx,
                y,
                _sentence_case(level_label(level)),
                ha="left",
                va="center",
                fontsize=style.FONT_SIZE_DENSE,
                color=color,
                fontweight="bold",
                transform=trans,
                rotation=BRACKET_NAME_ROTATION,
                rotation_mode="anchor",
            )


def _sentence_case(label: str) -> str:
    return label[0].upper() + label[1:]


def dominant_order(loadings: np.ndarray) -> np.ndarray:
    """Rows grouped by their largest-magnitude component, then by that loading."""
    dominant = np.abs(loadings).argmax(axis=1)
    value = np.abs(loadings[np.arange(len(loadings)), dominant])
    return np.lexsort((-value, dominant))


def label_heatmap_cells(
    ax: plt.Axes,
    entries: list[tuple[int, int, str]],
    shape: tuple[int, int],
    clear: bool = False,
) -> None:
    """Outline each (drawn row, component column, name) cell of a heatmap of
    `shape` and label it beside its column, toward the wider blank side:
    rightward up to the middle component, leftward after it. Labels are
    stacked `FACTOR_LABEL_GAP` apart across both sides, so two labels of one
    row never share a height. With `clear`, each label instead goes where it
    overlaps no outlined cell and no other label (`_clear_labels`)."""
    if clear:
        _clear_labels(ax, entries, shape)
        return
    n_rows, n_cols = shape
    gap = FACTOR_LABEL_GAP / (ax.bbox.height / ax.figure.dpi) * n_rows  # in rows
    entries = sorted(entries)
    heights = _spread_labels([row for row, _, _ in entries], gap, n_rows)
    for (row, col, name), y in zip(entries, heights, strict=True):
        _label_cell(ax, row, col, name, y, rightward=col < n_cols / 2)


def _overlaps(a, b, pad: float = 0.05) -> bool:
    return not (
        a[2] + pad <= b[0]
        or b[2] + pad <= a[0]
        or a[3] + pad <= b[1]
        or b[3] + pad <= a[1]
    )


def _label_text(ax, row: float, col: int, name: str, right: bool) -> tuple:
    """A label beside cell column `col` at height `row`: the text, its
    leader's start (the column edge) and end."""
    edge = col + (0.5 if right else -0.5)
    x = edge + (FACTOR_LEADER_LENGTH if right else -FACTOR_LEADER_LENGTH)
    text = ax.text(
        x,
        row,
        name,
        ha="left" if right else "right",
        va="center",
        fontsize=FACTOR_LABEL_FONT_SIZE,
    )
    return text, edge, x


def _data_extent(ax, text) -> tuple[float, float, float, float]:
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(
        text.get_window_extent(ax.figure.canvas.get_renderer()).get_points()
    )
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _is_clear(box, key, boxes: dict, placed: list, shape: tuple[int, int]) -> bool:
    """Whether a label's extent lies within the heatmap and clears every other
    outlined cell and every placed label."""
    low, high = -0.5 - 1e-6, np.array(shape[::-1]) - 0.5 + 1e-6
    if min(box[0], box[1]) < low or box[2] > high[0] or box[3] > high[1]:
        return False
    if any(_overlaps(box, b) for k, b in boxes.items() if k != key):
        return False
    return not any(_overlaps(box, q, 0.0) for q in placed)


def _candidates(prefer: bool, gap: float):
    """(right, dy) to try, in order: the side with room then the other, at the
    label's own height then up and down in half-gap steps."""
    for k in range(8):
        for dy in (0,) if k == 0 else (k * gap / 2, -k * gap / 2):
            for right in (prefer, not prefer):
                yield right, dy


def _place_label(ax, row, col, name, boxes, placed, shape) -> tuple:
    """The first clear place (`_candidates`, `_is_clear`): (text, leader start,
    end x, y, extent); when none is, the default place, kept within the
    heatmap's height, with no extent."""
    n_rows, n_cols = shape
    gap = FACTOR_LABEL_GAP / (ax.bbox.height / ax.figure.dpi) * n_rows
    prefer = col < n_cols / 2
    for right, dy in _candidates(prefer, gap):
        y = min(max(row + dy, 0), n_rows - 1)
        text, edge, x = _label_text(ax, y, col, name, right)
        box = _data_extent(ax, text)
        if _is_clear(box, (row, col), boxes, placed, shape):
            return text, edge, x, y, box
        text.remove()
    text, edge, x = _label_text(ax, row, col, name, prefer)
    _, y0, _, y1 = _data_extent(ax, text)
    y = row + max(-0.5 - y0, 0) - max(y1 - (n_rows - 0.5), 0)
    text.set_y(y)
    return text, edge, x, y, None


def _clear_labels(ax: plt.Axes, entries, shape: tuple[int, int]) -> None:
    """Outline each labeled cell, then place each label clear of every
    outlined cell and every label already placed (`_place_label`), on a
    leader line from its cell."""
    boxes = {
        (row, col): (col - 0.5, row - 0.5, col + 0.5, row + 0.5)
        for row, col, _ in entries
    }
    for row, col in boxes:
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=style.LINE_WIDTH_THIN,
            )
        )
    placed = []
    for row, col, name in sorted(entries):
        _, edge, x, y, box = _place_label(ax, row, col, name, boxes, placed, shape)
        ax.plot([edge, x], [row, y], color="black", lw=style.LINE_WIDTH_THIN)
        if box:
            placed.append(box)


def text_right_edge(ax) -> float:
    """The right edge of `ax`'s texts, in axes fraction (at least 1)."""
    renderer = ax.figure.canvas.get_renderer()
    edges = [
        ax.transAxes.inverted().transform(t.get_window_extent(renderer).get_points())[
            1, 0
        ]
        for t in ax.texts
    ]
    return max([1.0, *edges])


def _label_cell(
    ax: plt.Axes, row: int, col: int, name: str, y: float, rightward: bool
) -> None:
    """Outline one heatmap cell and label it at height `y` beside its column."""
    ax.add_patch(
        Rectangle(
            (col - 0.5, row - 0.5),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=style.LINE_WIDTH_THIN,
        )
    )
    edge = col + (0.5 if rightward else -0.5)
    x = edge + (FACTOR_LEADER_LENGTH if rightward else -FACTOR_LEADER_LENGTH)
    ax.plot([edge, x], [row, y], color="black", lw=style.LINE_WIDTH_THIN)
    ax.text(
        x,
        y,
        name,
        ha="left" if rightward else "right",
        va="center",
        fontsize=FACTOR_LABEL_FONT_SIZE,
    )


def _spread_labels(rows: list[float], gap: float, n_rows: int) -> np.ndarray:
    """Heights for sorted `rows` at least `gap` apart: each run of labels too
    close to fit is centered on its own rows, and kept within the heatmap."""
    clusters: list[list[float]] = []  # [first height, n labels, sum of rows]
    for row in rows:
        clusters.append([row, 1, row])
        while True:
            _, n, total = clusters[-1]
            top = min(max(total / n - (n - 1) * gap / 2, 0), n_rows - 1 - (n - 1) * gap)
            clusters[-1][0] = top
            if len(clusters) < 2 or clusters[-2][0] + clusters[-2][1] * gap <= top:
                break
            _, n_prev, total_prev = clusters.pop(-2)
            clusters[-1][1:] = [n + n_prev, total + total_prev]
    return np.concatenate(
        [first + gap * np.arange(n) for first, n, _ in clusters] or [np.empty(0)]
    )
