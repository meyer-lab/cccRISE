"""IBD draft: what the comparators recover as cell-type labels get finer
(`run_granularity_sampling`; labellings from `run_comparator_fits`).

x: the labelling's number of cell types -- the 7 lineages (fit under the
sampled levels' settings, ``_s0r0``), 12 / 24 / 36 fine types broken out of
their lineage at random (3 repeats each; points), and all 54 fine types.
Lines join the means.

A. Each cccRISE component's LR containment in each comparator's span (one
   panel per method, one line per component).
B. Per method: the share of its columns cccRISE's LR span contains
   (>= 0.85), the share whose best per-pair-min match is an LR match, and the
   share that are single-pair (effective pairs < 1.5).
C. Each method's selected rank (STACCato: its LR-mode rank).
D. Per labelling and method, the number of cccRISE components its LR span
   contains (>= 0.85) against its LR rank, coloured by the number of types:
   whether recovery follows the granularity or only the rank it brought."""

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

from ...runners.datasets import VALIDATION_DIR
from .. import common, style
from ..figureS_lr_concordance._common import METHODS, SHORT

DATASET = "ibd"
CONTAINED = 0.85
METHOD_COLORS = dict(zip(METHODS, style.CATEGORICAL[1:4], strict=True))


def load(name: str) -> pd.DataFrame:
    table = common.load_runner_output(
        f"granularity_sampling_{name}_{DATASET}.parquet",
        VALIDATION_DIR,
        hint=f"    uv run python -m cccrise.runners.run_granularity_sampling --dataset {DATASET}",
    )
    # the exact lineage fit is a reference; the curve uses the sampling protocol
    return table[table["labelling"] != "_coarse"]


def _jitter(x, seed=0):
    return x * np.exp(np.random.default_rng(seed).uniform(-0.03, 0.03, len(x)))


def _axis(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.grid(False)


def _containment(axes, comps, ticks):
    palette = sns.color_palette("Dark2", 8)
    for ax, method in zip(axes, METHODS, strict=True):
        sub = comps[comps["method"] == method]
        for k, part in sub.groupby("cccrise_component"):
            color = palette[k - 1]
            ax.scatter(
                _jitter(part["n_types"].to_numpy(float), k), part["lr_containment"],
                s=style.POINT_SIZE_BACKGROUND, color=color, alpha=0.6, lw=0,
            )  # fmt: skip
            means = part.groupby("n_types")["lr_containment"].mean()
            ax.plot(
                means.index,
                means.values,
                color=color,
                lw=style.LINE_WIDTH,
                label=f"C{k}",
            )
        _axis(ax, ticks)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(SHORT[method], fontsize=style.FONT_SIZE_TITLE)
        ax.set_xlabel("cell types")
    axes[0].set_ylabel("cccRISE component's LR\ncontainment in comparator span")
    axes[-1].legend(fontsize=style.FONT_SIZE_DENSE - 1, ncol=2, loc="center right")


def _columns(axes, cols, ticks):
    measures = {
        "in cccRISE's span": cols["reverse_containment"] >= CONTAINED,
        "LR match": cols["lr_match"],
        "single-pair": cols["single_pair"],
    }
    for ax, (label, flag) in zip(axes, measures.items(), strict=True):
        shares = (
            cols.assign(flag=flag)
            .groupby(["method", "labelling", "n_types"])["flag"]
            .mean()
            .reset_index()
        )
        for method, part in shares.groupby("method"):
            color = METHOD_COLORS[method]
            ax.scatter(
                _jitter(part["n_types"].to_numpy(float)),
                part["flag"],
                s=style.POINT_SIZE_BACKGROUND,
                color=color,
                alpha=0.6,
                lw=0,
            )
            means = part.groupby("n_types")["flag"].mean()
            ax.plot(
                means.index,
                means.values,
                color=color,
                lw=style.LINE_WIDTH,
                label=SHORT[method],
            )
        _axis(ax, ticks)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(label, fontsize=style.FONT_SIZE_DENSE + 1)
        ax.set_xlabel("cell types")
    axes[0].set_ylabel("share of comparator columns")
    axes[0].legend(fontsize=style.FONT_SIZE_DENSE - 1)


def _lr_rank(table):
    """One row per (method, labelling) with its LR rank as ``r``."""
    first = table.drop_duplicates(["method", "labelling"])
    parts = first["rank"].str.split("x")
    stac = first["method"] == "STACCato"
    return first.assign(r=parts.str[0].where(~stac, parts.str[1]).astype(int))


def _ranks(ax, cols, ticks):
    first = _lr_rank(cols)
    for method, part in first.groupby("method"):
        color = METHOD_COLORS[method]
        ax.scatter(
            _jitter(part["n_types"].to_numpy(float)),
            part["r"],
            s=style.POINT_SIZE_BACKGROUND,
            color=color,
            alpha=0.6,
            lw=0,
        )
        means = part.groupby("n_types")["r"].mean()
        ax.plot(
            means.index,
            means.values,
            color=color,
            lw=style.LINE_WIDTH,
            label=SHORT[method],
        )
    _axis(ax, ticks)
    ax.set_title("LR rank", fontsize=style.FONT_SIZE_DENSE + 1)
    ax.set_xlabel("cell types")


def _recovery_by_rank(ax, comps, ticks):
    contained = (
        comps.assign(hit=comps["lr_containment"] >= CONTAINED)
        .groupby(["method", "labelling"])["hit"]
        .sum()
        .rename("hits")
    )
    table = _lr_rank(comps).join(contained, on=["method", "labelling"])
    norm = mpl.colors.Normalize(min(ticks), max(ticks))
    handles = []
    for marker, (method, part) in zip("osD", table.groupby("method"), strict=True):
        points = ax.scatter(
            _jitter(part["r"].to_numpy(float), len(part)),
            part["hits"] + _jitter(np.ones(len(part)), len(part) + 1) - 1,
            c=part["n_types"], cmap=style.CMAPS["sequential"], norm=norm,
            marker=marker, s=style.POINT_SIZE, lw=0,
        )  # fmt: skip
        handles.append(
            mpl.lines.Line2D(
                [], [], ls="", marker=marker, color=style.COLORS["reference"],
                label=SHORT[method],
            )
        )  # fmt: skip
    ax.figure.colorbar(points, ax=ax, ticks=ticks, format="%d", label="cell types")
    ax.set_xlabel("LR rank")
    ax.set_title("components contained", fontsize=style.FONT_SIZE_DENSE + 1)
    ax.legend(handles=handles, fontsize=style.FONT_SIZE_DENSE - 1, loc="lower right")
    ax.grid(False)


def plot(fig):
    comps, cols = load("components"), load("columns")
    ticks = sorted(comps["n_types"].unique())
    top, bottom = fig.subfigures(2, 1, height_ratios=(1, 1))
    _containment(top.subplots(1, len(METHODS), sharey=True), comps, ticks)
    axes = bottom.subplots(1, 5)
    _columns(axes[:3], cols, ticks)
    _ranks(axes[3], cols, ticks)
    _recovery_by_rank(axes[4], comps, ticks)


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 130 * style.MM))
