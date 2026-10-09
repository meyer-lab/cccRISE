"""Which cccRISE components the comparators recover, component-blind on the
comparator side: how much of each cccRISE LR column lies in the span of each
comparator fit's LR columns (``||P_X a_k||^2`` for unit `a_k`; 1 when some
combination of the fit's columns reproduces it). Rows: own-rank fits on the
coarse and fine annotation; columns: cccRISE components
(`cccrise-comparator-agreement`). `plot_reverse`: the other direction, each
comparator column's containment in the span of cccRISE's LR columns."""

import numpy as np
import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from .. import common, style
from ._common import ANNOTATIONS, METHODS, SHORT

SUFFIXES = {"coarse": "", "fine": "_fine"}
FITS = [(m, a) for a in ANNOTATIONS for m in METHODS]


def load(direction: str = "cccRISE in comparator") -> pd.DataFrame:
    """LR-mode containment of every cccRISE component in every own-rank fit
    (or, ``"comparator in cccRISE"``, of every fit's columns in cccRISE)."""
    frames = [
        common.load_runner_output(
            f"comparator_containment_alad{SUFFIXES[a]}.parquet",
            VALIDATION_DIR,
            hint="    uv run cccrise-comparator-agreement --dataset alad",
        ).assign(annotation=a)
        for a in ANNOTATIONS
    ]
    table = pd.concat(frames, ignore_index=True)
    return (
        table[
            (table["direction"] == direction)
            & (table["mode"] == "lr")
            & (table["rank_mode"] == "own")
        ]
        .set_index(["method", "annotation", "component"])
        .sort_index()
    )


def _grid(table, comps, column):
    return np.array([[table.loc[(*f, c), column] for c in comps] for f in FITS])


def _annotate(ax, values, size=style.FONT_SIZE_DENSE):
    for (i, j), v in np.ndenumerate(values):
        if np.isfinite(v):
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=size,
                color="white" if v > 0.6 else style.COLORS["ink"],
            )  # fmt: skip


def _colorbar(fig, image, ax, label):
    fig.colorbar(
        image, ax=ax, label=label, location="bottom", shrink=0.8, pad=0.02
    )  # fmt: skip


def plot(fig):
    table = load()
    comps = sorted(table.index.get_level_values("component").unique())
    values = _grid(table, comps, "containment")
    ax = fig.subplots()
    image = ax.imshow(
        values, cmap=style.CMAPS["sequential"], vmin=0, vmax=1, aspect="auto"
    )
    _annotate(ax, values)
    ax.set_yticks(range(len(FITS)), [f"{SHORT[m]} {a}" for m, a in FITS])
    ax.set_xticks(range(len(comps)), [f"C{c}" for c in comps])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    ax.axhline(len(METHODS) - 0.5, color="white", lw=3)
    ax.set_xlabel("cccRISE component")
    ax.set_title("Recovery of cccRISE components", fontsize=style.FONT_SIZE_TITLE)
    ax.grid(False)
    _colorbar(fig, image, ax, "LR containment in comparator span")


REVERSE_ROWS = 40  # grid rows the tallest fit column spans


def _fit_column(ax, values):
    """One fit's columns as a single heatmap column, numbered from 1."""
    image = ax.imshow(
        values[:, None], cmap=style.CMAPS["sequential"], vmin=0, vmax=1,
        aspect="auto",
    )  # fmt: skip
    _annotate(ax, values[:, None], size=style.FONT_SIZE_DENSE - 1.5)
    ax.set_yticks(
        range(len(values)), [str(i + 1) for i in range(len(values))],
        fontsize=style.FONT_SIZE_DENSE - 1.5,
    )  # fmt: skip
    ax.set_xticks([])
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    return image


def plot_reverse(fig):
    """One column per fit, coarse fits then fine. A fit's column is as tall
    as the square root of its number of columns allows (relative to the
    largest), so 3 and 10 columns end up at similar heights without the
    short ones stretched to full height."""
    fig = fig.subfigures(1, 1)  # one level down: the panel letter takes the top one
    table = load("comparator in cccRISE")["containment"]
    counts = {fit: len(table.loc[fit]) for fit in FITS}
    grid = fig.add_gridspec(
        REVERSE_ROWS + 5, len(FITS) + 1, width_ratios=(1, 1, 1, 0.3, 1, 1, 1)
    )
    for i, fit in enumerate(FITS):
        col = i + (i >= len(METHODS))  # skip the spacer between annotations
        rows = round(REVERSE_ROWS * np.sqrt(counts[fit] / max(counts.values())))
        ax = fig.add_subplot(grid[:rows, col])
        image = _fit_column(ax, table.loc[fit].sort_index().to_numpy())
        ax.set_title(SHORT[fit[0]], fontsize=style.FONT_SIZE_DENSE + 0.5)
        if fit[0] == METHODS[1]:
            ax.annotate(
                fit[1], (0.5, 1), xytext=(0, 22), xycoords="axes fraction",
                textcoords="offset points", ha="center",
                fontsize=style.FONT_SIZE_DENSE + 1,
            )  # fmt: skip
    fig.colorbar(
        image, cax=fig.add_subplot(grid[-2:, 1:-1]), orientation="horizontal",
        label="LR containment in cccRISE span", ticks=[0, 0.5, 1],
    )  # fmt: skip
    fig.suptitle(
        "Recovery of comparator columns", fontsize=style.FONT_SIZE_TITLE,
        fontweight="normal",
    )  # fmt: skip


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE / 2, 85 * style.MM))
