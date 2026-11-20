"""Figure 5's shared drawing helpers: colorbars and heatmap
cells (`_cell`), each component's best single column in every fit
(`best_column_grid`, Figure 5A and the IBD supplement) and the axes of a
pairs-by-columns heatmap (`_pair_axes`, `_split`)."""

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from .. import common, style
from ._common import DATASET_COLORS, DATASET_LABELS
from ._programs import FITS, best_columns

HEAT = style.CMAPS["weight"]
#: Red marks tying panels to the missed supplements (`_missed`).
LINK = "#d00000"
#: Gap between a heatmap's blocks of columns, in columns.
BLOCK_GAP = 0.3
LOADING_LABEL = "|Loading| / max |loading|"
COS2_LABEL = "Best-match cos$^2$"
LABEL_SIZE = style.FONT_SIZE_DENSE - 0.7


def colorbar_under(
    ax, label: str, drop: float, x0=0.2, width=0.6, thickness=0.04
) -> None:
    """A horizontal colorbar `drop` of `ax`'s height below it."""
    cax = ax.inset_axes((x0, -drop, width, thickness))
    ax.figure.colorbar(
        ScalarMappable(Normalize(0, 1), HEAT), cax=cax, orientation="horizontal"
    )
    cax.set_xticks([0, 1])
    cax.tick_params(labelsize=style.FONT_SIZE_DENSE, length=1.5, pad=1)
    cax.set_xlabel(label, fontsize=style.FONT_SIZE_DENSE, labelpad=0)


#: `colorbar_beneath`: the bar's top below the axes, and its thickness (mm)
BENEATH_DROP, BENEATH_THICKNESS = 5.2, 1.6


def colorbar_beneath(ax, label: str) -> None:
    """`colorbar_under` a `fit_grid`'s fit labels at a fixed distance and
    thickness (mm), whatever the grid's height."""
    height = ax.bbox.height / ax.figure.dpi / style.MM
    colorbar_under(
        ax,
        label,
        (BENEATH_DROP + BENEATH_THICKNESS) / height,
        thickness=BENEATH_THICKNESS / height,
    )


def _cell(ax, x: float, y: float, value: float, text: bool = False) -> None:
    ax.add_patch(Rectangle((x, y), 1, 1, fc=HEAT(value), ec="black", lw=0.3))
    if text:
        ax.text(
            x + 0.5,
            y + 0.5,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
            color="white" if value > 0.6 else style.COLORS["ink"],
        )


def _clean_heatmap(ax) -> None:
    ax.tick_params(pad=1.5)


def _cohort_labels(ax, centers: dict) -> None:
    """Each cohort's name beside a bar in its color, right of `ax`."""
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    width = ax.bbox.width / ax.figure.dpi
    bar, text = 1 + 1.2 * style.MM / width, 1 + 2.4 * style.MM / width
    for dataset, y in centers.items():
        ax.plot(
            [bar, bar],
            [y[0] - 0.4, y[-1] + 0.4],
            color=DATASET_COLORS[dataset],
            lw=2.5,
            transform=trans,
            clip_on=False,
            solid_capstyle="butt",
        )
        ax.text(
            text,
            y.mean(),
            DATASET_LABELS[dataset],
            rotation=-90,
            ha="left",
            va="center",
            fontsize=style.FONT_SIZE_LABEL,
            transform=trans,
        )


def fit_grid(ax, grids: dict[str, pd.DataFrame], bars: bool = True) -> None:
    """A components-by-fits heatmap with values, each cohort's components
    (`grids`, indexed by component, columns (method, labelling)) under the
    last, fits in `FITS` order, method names above and (if `bars`) cohort bars
    right."""
    y, ticks, names, centers = 0.0, [], [], {}
    for dataset, grid in grids.items():
        centers[dataset] = y + np.arange(len(grid)) + 0.5
        for component, row in grid.iterrows():
            for j, fit in enumerate(FITS):
                _cell(ax, j, y, row[fit], text=True)
            ticks.append(y + 0.5)
            names.append(f"C{component}")
            y += 1
        y += 0.5
    ax.set(xlim=(0, len(FITS)), ylim=(y - 0.5, 0))
    ax.set_yticks(ticks, names)
    ax.set_xticks(np.arange(len(FITS)) + 0.5, [lab[0].upper() for _, lab in FITS])
    ax.tick_params(labelsize=style.FONT_SIZE_DENSE)
    _clean_heatmap(ax)
    for i, method in enumerate(dict.fromkeys(m for m, _ in FITS)):
        ax.text(
            2 * i + 1,
            -0.25,
            method,
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
        )
    if bars:
        _cohort_labels(ax, centers)


def best_column_grid(ax, datasets=tuple(DATASET_LABELS), bars: bool = True) -> None:
    """Each program's best single column in each fit, the cohorts of
    `datasets` one under another (ALAD over IBD); no colorbar. `bars` is
    `fit_grid`'s."""
    t = pd.concat([best_columns(d).assign(Dataset=DATASET_LABELS[d]) for d in datasets])
    common.source_data(
        "program_best_column_cos2",
        t.rename(columns={"Cos2": "Best-match cos2"}),
        "Each program's largest cos^2 with one LR column of each comparator fit.",
    )
    fit_grid(
        ax,
        {
            dataset: t[t["Dataset"] == label].pivot_table(
                index="Program", columns=["Method", "Labelling"], values="Cos2"
            )
            for dataset, label in DATASET_LABELS.items()
            if dataset in datasets
        },
        bars,
    )


def _pair_axes(ax, rows, spans, end: float, ticks, heads) -> None:
    positions, labels, bold = ticks
    ax.set(xlim=(0, end), ylim=(max(s[1] for s in spans.values()), 0))
    ax.set_yticks(
        [y + 0.5 for *_, y in rows],
        [pair.replace("^", "–").replace("&", "+") for _, pair, _ in rows],
        fontsize=style.FONT_SIZE_DENSE - 0.7,
    )
    ax.set_xticks(positions, labels, fontsize=style.FONT_SIZE_DENSE - 0.7)
    for i in set(bold):
        ax.get_xticklabels()[i].set_fontweight("bold")
    _clean_heatmap(ax)
    for name, x in heads.items():
        ax.text(x, -0.4, name, ha="center", va="bottom", fontsize=style.FONT_SIZE_DENSE)
    for program, (y0, y1) in spans.items():
        ax.plot(
            [end + 0.25] * 2, [y0 + 0.1, y1 - 0.1], color="0.5", lw=0.6, clip_on=False
        )
        ax.text(
            end + 0.5,
            (y0 + y1) / 2,
            f"C{program}",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            clip_on=False,
        )
