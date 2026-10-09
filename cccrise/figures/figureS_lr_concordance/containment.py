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


def _reverse_rows(table, fits):
    """Each of `fits`' columns in order, a blank (NaN) row between fits: the
    column of values, the fit of every row (None for blanks) and the tick
    labels."""
    values, owners, labels = [], [], []
    for i, fit in enumerate(fits):
        if i:
            values, owners, labels = [*values, np.nan], [*owners, None], [*labels, ""]
        sub = table.loc[fit, "containment"].sort_index()
        values += list(sub)
        owners += [fit] * len(sub)
        labels += [str(c) for c in sub.index]
    return np.array(values)[:, None], owners, labels


def blocked_heatmap(ax, values, labels, groups, length, group_x=-0.25):
    """Containment values (NaN: blank) with small numbers, rows padded with
    blanks to `length` rows, each row's tick label in `labels` and each group
    of rows (`groups`, None for blanks) named left of its ticks."""
    pad = np.full((length - len(values), values.shape[1]), np.nan)
    values = np.vstack([values, pad])
    cmap = style.CMAPS["sequential"].with_extremes(bad="white")
    image = ax.imshow(
        np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=1, aspect="auto"
    )
    _annotate(ax, values, size=style.FONT_SIZE_DENSE - 1.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=style.FONT_SIZE_DENSE - 1.5)
    for group in dict.fromkeys(g for g in groups if g):
        rows = [i for i, g in enumerate(groups) if g == group]
        ax.text(
            group_x, np.mean(rows), group, ha="right", va="center",
            transform=ax.get_yaxis_transform(), fontsize=style.FONT_SIZE_DENSE,
        )  # fmt: skip
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    return image


def _reverse_column(ax, table, fits, length, name_groups):
    """One annotation's comparator columns as a single heatmap column,
    padded with blank rows to `length` so both annotations share a scale;
    the methods are named only when `name_groups`."""
    values, owners, labels = _reverse_rows(table, fits)
    groups = [SHORT[f[0]] if f and name_groups else None for f in owners]
    image = blocked_heatmap(ax, values, labels, groups, length)
    ax.set_xticks([])
    ax.set_title(fits[0][1], fontsize=style.FONT_SIZE_DENSE + 1)
    return image


def plot_reverse(fig):
    """Coarse fits' columns beside the fine fits' columns."""
    fig = fig.subfigures(1, 1)  # one level down: the panel letter takes the top one
    table = load("comparator in cccRISE")
    groups = [[f for f in FITS if f[1] == a] for a in ANNOTATIONS]
    length = max(len(_reverse_rows(table, g)[0]) for g in groups)
    axes = fig.subplots(1, 5, width_ratios=(0.9, 1, 0.25, 1, 0.9))
    for ax in axes[::4]:
        ax.set_axis_off()
    axes[2].set_axis_off()
    for ax, fits, first in zip(axes[1::2], groups, (True, False), strict=True):
        image = _reverse_column(ax, table, fits, length, name_groups=first)
    fig.suptitle(
        "Recovery of comparator columns", fontsize=style.FONT_SIZE_TITLE,
        fontweight="normal",
    )  # fmt: skip
    fig.colorbar(
        image, ax=list(axes[1:4]), label="LR containment in cccRISE span",
        location="bottom", ticks=[0, 0.5, 1], pad=0.02, shrink=0.8,
    )  # fmt: skip


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE / 2, 85 * style.MM))
