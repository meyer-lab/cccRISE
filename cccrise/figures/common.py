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
"""

import functools
import importlib
import itertools
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import ascii_letters, ascii_uppercase
from typing import Any

import anndata
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text

matplotlib.use("AGG")

from ..anova import (
    get_sample_to_group,
    prepare_group_labels,
)
from ..factors import (
    ALAD_FACTORS_PATH,
    COVID19_FACTORS_PATH,
    DATA_DIR,
    clean_memory,
    load_factors_to_anndata,
    load_or_compute_dataframe,
)
from ..import_data import import_alad, import_balf_covid
from ..runners.datasets import DATASETS
from ..runners.provenance import read_metadata
from ..utils import (
    correct_conditions,
)
from . import style

__all__ = [
    "ALAD_FACTORS_PATH",
    "COVID19_FACTORS_PATH",
    "DATA_DIR",
    "clean_memory",
    "genFigure",
    "getSetup",
    "get_alad_data",
    "get_covid_data",
    "load_factors_to_anndata",
    "load_or_compute_dataframe",
    "load_runner_output",
    "subplotLabel",
]


def get_covid_data(
    project_cells: bool = False, do_embedding: bool = False
) -> anndata.AnnData:
    """Load COVID dataset with factors and optional single-cell projections."""
    if not project_cells:
        return load_factors_to_anndata(COVID19_FACTORS_PATH)
    raw = import_balf_covid(gene_threshold=0.001, normalize=True)
    return load_factors_to_anndata(
        COVID19_FACTORS_PATH, adata=raw, do_embedding=do_embedding
    )


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
            "    uv run cccrise-cpdrank  --dataset <dataset> --rise-rank <N>\n"
            "    uv run cccrise-stability --dataset <dataset> --rise-rank <N> --cp-rank <K>"
        )
        raise FileNotFoundError(
            f"{path} not found. Produce it with the matching runner, e.g.\n{hint}"
        )
    return pd.read_parquet(path)


def selected_ranks(dataset: str) -> tuple[int, int]:
    """(RISE rank, CP rank) of `dataset`'s canonical fit, from its provenance."""
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
    # Measured at the save dpi: text hinting makes low-dpi extents narrower.
    fig = plt.figure(
        figsize=(FIXED_MEASURE_SIZE, FIXED_MEASURE_SIZE),
        dpi=plt.rcParams["savefig.dpi"],
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
    """Letter each axes A, B, C, ... at its left margin and its row's top,
    tagged with its panel module's name for `record_panel_letters`."""
    width, height = fig.get_size_inches()
    for letter, (box, x, _, label_y) in zip(ascii_uppercase, placements, strict=False):
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
            if isinstance(box, _Stack):
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


def _draw_row(fig: Figure, row: list) -> list:
    """The boxes of a row's modules, each list of modules one `_Stack`."""
    boxes = []
    for item in row:
        if isinstance(item, list):
            boxes.append(_Stack([b for m in item for b in _draw_fixed(fig, m)]))
        else:
            boxes += _draw_fixed(fig, item)
    return boxes


def panel_axes(fig: Figure, panel: str) -> plt.Axes:
    """The axes `compose_fixed` drew panel module `panel` (its name) into,
    for adjustments spanning panels once the figure is laid out."""
    return next(ax for ax in fig.axes if ax.get_label() == panel)


def _draw_fixed(fig: Figure, module) -> list[_AxesBox]:
    """Draw `module` into its fixed-size axes and measure their margins."""
    boxes = []
    for kind in module.AXES:
        w, h = style.AXIS_SIZE[kind]
        ax = fig.add_axes((0.4, 0.4, w / FIXED_MEASURE_SIZE, h / FIXED_MEASURE_SIZE))
        ax.set_label(module.__name__.rsplit(".", 1)[-1])  # found by `panel_axes`
        boxes.append(_AxesBox(ax, w, h))
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


def colorbar(ax: plt.Axes, mappable, label: str):
    """A colorbar beside `ax`, in its margin rather than taken from its width."""
    width = ax.bbox.width / ax.figure.dpi
    cax = ax.inset_axes(
        (1 + style.COLORBAR_PAD / width, 0, style.COLORBAR_WIDTH / width, 1)
    )
    return ax.figure.colorbar(mappable, cax=cax, label=label)


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


def build_figure(figure: str, panel: str | None = None) -> Figure:
    """Figure `figure` (e.g. '2', 'S_cmp_stability'), or one of its panels."""
    style.apply_style()
    package = f"figure{figure}"
    # Supplements may be nested in their main figure's package
    (master,) = FIGURES_DIR.glob(f"**/{package}/{package}.py")
    parts = master.parent.relative_to(FIGURES_DIR).parts
    module = importlib.import_module(
        ".".join(("cccrise.figures", *parts, panel or package))
    )
    return module.makeFigure()


def genFigure():
    """Build a figure, or one of its panels, into output/figure<N>/:

    uv run cccrise-fbuild 2                      # figures/figure2/figure2.py
    uv run cccrise-fbuild 2 rise_rank_selection  # figures/figure2/rise_rank_selection.py
    uv run cccrise-fbuild 2 --png                # PNG instead of SVG
    uv run cccrise-fbuild S_cmp_stability        # figures/figure2/figureS_cmp_stability/
    """
    start = time.time()
    args = [arg for arg in sys.argv[1:] if arg != "--png"]
    extension = "png" if "--png" in sys.argv else "svg"
    figure, panel = args[0], (args[1] if len(args) > 1 else None)

    fig = build_figure(figure, panel)
    output_dir = OUTPUT_DIR / f"figure{figure}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": figures are drawn at their final print size.
    fig.savefig(output_dir / f"{panel or f'figure{figure}'}.{extension}")
    if panel is None:
        record_panel_letters(figure, fig)
    print(f"Figure {' '.join(args)} is done after {time.time() - start:.1f} s.\n")


def record_panel_letters(figure: str, fig: Figure) -> None:
    """Set `fig<figure>` in `PANELS_PATH` to the letters of each panel
    module's axes, as tagged by `_letter_axes` or `label_panels`."""
    letters: dict[str, list[str]] = {}
    for text in fig.findobj(Text):
        if module := text.get_gid():
            letters.setdefault(module, []).append(text.get_text().lower())
    panels = yaml.safe_load(PANELS_PATH.read_text()) or {}
    panels[f"fig{figure}"] = letters
    PANELS_PATH.write_text(
        yaml.safe_dump(panels, sort_keys=False, default_flow_style=None)
    )


# =============================================================================
# Shared panels
# =============================================================================

SIG_ALPHA = 0.05


RANK_LINE: dict[str, Any] = {
    "color": style.COLORS["reference"],
    "linestyle": style.LINE_STYLE_REFERENCE,
    "linewidth": style.LINE_WIDTH_THIN,
}


def mark_rank(ax: plt.Axes, rank: int | None) -> None:
    """Vertical reference line at the rank the canonical fit uses."""
    if rank is not None:
        ax.axvline(rank, zorder=0, **RANK_LINE)


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


# --- Rank selection (ALAD Fig 2, IBD Figs 4-5) --------------------------------


def plot_rise_rank(sweep: pd.DataFrame, ax: plt.Axes, rank: int | None = None):
    """In-sample R2X of the full-data fit, and in-sample (training block) and
    held-out BiCV R2X as mean +/- SD over the BiCV repeats, against RISE
    rank, from `cccrise-riserank`. BiCV peaks or plateaus near the rank that
    generalizes to unseen cells and genes."""
    full = sweep[sweep["Metric"] == "Fit R2X"].groupby("Rank")["R2X"].mean()
    ax.plot(full.index, full, color=style.COLORS["fit_full"], label="Fit")
    repeats = sweep[sweep["Metric"] == "BiCV R2X"].groupby("Rank")
    for column, label, color in (
        ("Train Block R2X", "BiCV train", style.COLORS["fit"]),
        ("R2X", "BiCV test", style.COLORS["heldout"]),
    ):
        mean, sd = repeats[column].mean(), repeats[column].std().fillna(0)
        plot_mean_sd(ax, mean.index, mean, sd, color, label)
    mark_rank(ax, rank)
    ax.set_xlabel("RISE rank")
    ax.set_ylabel("$R^2X$")
    ax.legend()


def plot_mean_sd(ax: plt.Axes, x, mean, sd, color: str, label: str, **kwargs):
    """`mean` against `x` with a shaded, thinly outlined +/- `sd` band;
    `kwargs` go to the mean line."""
    ax.plot(x, mean, color=color, label=label, **kwargs)
    for bound in (mean - sd, mean + sd):
        ax.plot(x, bound, color=color, linewidth=style.LINE_WIDTH_THIN)
        ax.fill_between(x, mean, bound, color=color, alpha=0.25, lw=0)


def plot_cpd_fit_r2x(
    sweep: pd.DataFrame,
    ax: plt.Axes,
    rank: int | None = None,
    color: str = style.COLORS["fit"],
    marker: str | None = "o",
):
    """Full-tensor CP fit R2X against CP rank, from `cccrise-cpdrank`."""
    ax.plot(sweep["Rank"], sweep["FitR2X"], marker=marker, color=color, label="Fit")
    mark_rank(ax, rank)
    ax.set_xlabel("CP rank")
    ax.set_ylabel("Fit $R^2X$")


def plot_collinearity(
    sweep: pd.DataFrame,
    ax: plt.Axes,
    rank: int | None = None,
    colors: tuple[str, str, str] = (
        style.COLORS["muted"],
        style.COLORS["emphasis"],
        style.COLORS["up"],
    ),
):
    """Number of modes the most collinear component pair is collinear in, per
    CP rank, marked by whether that pair duplicates or cancels itself. A
    canceling pair (collinear in >= 2 modes with opposing signs) flags a
    degenerate fit. `colors` is (no collinear pair, near-duplicate,
    canceling)."""
    none_color, duplicate_color, canceling_color = colors
    ax.plot(
        sweep["Rank"],
        sweep["MaxModesCollinear"],
        color=style.COLORS["muted"],
        zorder=1,
    )
    # NaN = no collinear pair; `fillna` keeps those rows out of both masks.
    flag = sweep["WorstPairCanceling"]
    none_found = flag.isna()
    canceling = flag.fillna(False).astype(bool) & ~none_found
    for mask, color, marker, label in (
        (none_found, none_color, "o", "None collinear"),
        (~canceling & ~none_found, duplicate_color, "o", "Near-duplicate"),
        (canceling, canceling_color, "D", "Canceling"),
    ):
        ax.scatter(
            sweep.loc[mask, "Rank"],
            sweep.loc[mask, "MaxModesCollinear"],
            color=color,
            marker=marker,
            s=style.POINT_SIZE,
            label=label,
            zorder=2,
        )
    mark_rank(ax, rank)
    ax.set_ylim(-0.3, 7.5)  # headroom for the legend
    ax.set_yticks(range(5))
    ax.set_xlabel("CP rank")
    ax.set_ylabel("Modes collinear")
    ax.legend(loc="upper right", fontsize=style.FONT_SIZE_DENSE)


# --- Outcome associations (ALAD Fig 3, IBD Fig 5) ------------------------------

#: Outcome levels in reading order (healthy/control first); unlisted levels
#: follow alphabetically.
LEVEL_ORDER = [
    "control",
    "declined",
    "recovered",
    "No CLAD/Stable",
    "No CLAD/Recovered",
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
    xlabel: str = "CP component",
    p_column: str = "PAdj",
    p_label: str = "FDR",
) -> None:
    """Eta-squared of each component (columns) against each outcome
    association (rows), both in `table` order, from
    `anova.run_multiple_factors_anova`; bold with a star where `p_column` is
    below `SIG_ALPHA`."""
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
    for (i, j), value in np.ndenumerate(values):
        if not np.isfinite(value):
            continue
        significant = padj.iloc[i, j] < SIG_ALPHA
        ax.text(
            j,
            i,
            f"{value:.2f}" + ("*" if significant else ""),
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold" if significant else "normal",
            color="white" if value > 0.6 * vmax else style.COLORS["ink"],
        )
    ax.set_xticks(range(effect.shape[1]), effect.columns)
    ax.set_yticks(range(len(rows)), rows, fontsize=style.FONT_SIZE_DENSE)
    ax.set_xlabel(xlabel)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    colorbar(ax, image, rf"$\eta^2$ (* {p_label} < {SIG_ALPHA})")


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
    ax.set_xlabel("CP component")
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
LEVEL_LABELS = {"No CLAD/Stable": "No CLAD", "No CLAD/Recovered": "Recovered"}


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
        fontsize=style.FONT_SIZE_DENSE,
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
    ax.set(xlabel="RISE eigenstate", ylabel="Loading")
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
    components = np.arange(loadings.shape[1])
    ax.set_xticks(components, components + 1)
    ax.set_xlabel(xlabel)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    return image


def dominant_order(loadings: np.ndarray) -> np.ndarray:
    """Rows grouped by their largest-magnitude component, then by that loading."""
    dominant = np.abs(loadings).argmax(axis=1)
    value = np.abs(loadings[np.arange(len(loadings)), dominant])
    return np.lexsort((-value, dominant))


def label_heatmap_cells(
    ax: plt.Axes, entries: list[tuple[int, int, str]], shape: tuple[int, int]
) -> None:
    """Outline each (drawn row, component column, name) cell of a heatmap of
    `shape` and label it beside its column, toward the wider blank side:
    rightward up to the middle component, leftward after it, each side's
    labels stacked `FACTOR_LABEL_GAP` apart."""
    n_rows, n_cols = shape
    gap = FACTOR_LABEL_GAP / (ax.bbox.height / ax.figure.dpi) * n_rows  # in rows
    for rightward in (True, False):
        side = sorted(e for e in entries if (e[1] <= n_cols // 2) == rightward)
        heights = _spread_labels([row for row, _, _ in side], gap, n_rows)
        for (row, col, name), y in zip(side, heights, strict=True):
            _label_cell(ax, row, col, name, y, rightward)


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
    """Heights for sorted `rows` at least `gap` apart, centered on them and
    kept within the heatmap."""
    placed = np.array(rows, dtype=float)
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    if len(placed):
        placed -= placed.mean() - np.mean(rows)
        placed -= max(placed[-1] - (n_rows - 1), 0) + min(placed[0], 0)
    return placed


# =============================================================================
# Legacy layout helpers, used only by figures/legacy (including
# legacy/commonFuncs); removed with them.
# =============================================================================


def getSetup(
    figsize: tuple[float, float] | tuple[int, int], gridd: tuple[int, int]
) -> tuple[list[plt.Axes], Figure]:
    """Establish figure set-up with subplots."""
    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, layout="constrained")
    gs1 = gridspec.GridSpec(gridd[0], gridd[1], figure=f)

    # Get list of axis objects
    ax = [f.add_subplot(gs1[x]) for x in range(gridd[0] * gridd[1])]

    return ax, f


def subplotLabel(axs: list[plt.Axes]):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_letters[ii],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )
