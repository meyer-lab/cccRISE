"""Supplementary figure: how each comparator selected its rank, per dataset
and annotation (rows), from the selection traces its own-rank fit keeps
(`benchmark.fit_comparators`; `benchmark_fits_<dataset><suffix>.nc`).

- Pseudobulk CPD: cccRISE's rank rule (`rank_selection.select_rank_by_rule`).
  The share of its production starts passing the collinearity screen by
  rank, with the 90% line; open markers are ranks failing the rule on core
  consistency or agreement despite the screen; the rank is the largest
  passing one. Sweeps stopped at their first failing rank
  (`pb_stop_at_failure`) are short.
- Tensor-cell2cell: mean normalised reconstruction error by rank; the rank
  is the elbow (kneedle).
- STACCato: the leading eigenvalue shares of each mode's unfolding (log),
  with the 1% line; each mode's rank is the number above it.

A row whose fits have not been run is left blank."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib.ticker import MaxNLocator

from ...benchmark import STACCATO_VAREXP, read_fits
from ...rank_selection import RULE_PASS_SHARE
from ...runners.datasets import DATASETS, VALIDATION_DIR
from .. import common, style

ROWS = (
    ("alad", "", "ALAD coarse"),
    ("alad", "_fine", "ALAD fine"),
    ("ibd", "", "IBD lineage"),
    ("ibd", "_fine", "IBD fine"),
)
METHODS = ("Pseudobulk CPD", "Tensor-cell2cell", "STACCato")
STAC_MODES = ("sample", "LR", "sender", "receiver")
COLOR = style.COLORS["emphasis"]


def _fits(dataset, suffix):
    path = VALIDATION_DIR / DATASETS[dataset].benchmark_fits_filename(suffix)
    if not Path(path).exists():
        return None, None
    fits, info, _ = read_fits(Path(path))
    return {f.method: f for f in fits if f.rank_mode == "own"}, info


def _selected(ax, rank):
    ax.axvline(rank, color=style.COLORS["reference"], ls=":", lw=style.LINE_WIDTH_THIN)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _pb(ax, fit):
    share = np.asarray(fit.details["rule_pass_share"], float)
    passes = np.asarray(fit.details["rule_passes"], bool)
    ranks = np.arange(1, len(share) + 1)
    other = (share >= RULE_PASS_SHARE) & ~passes  # failed on CORCONDIA or agreement
    ax.plot(ranks, share, color=COLOR, lw=style.LINE_WIDTH, zorder=1)
    ax.scatter(
        ranks, share, s=style.POINT_SIZE, zorder=2, lw=0.8, edgecolor=COLOR,
        facecolor=[("white" if o else COLOR) for o in other],
    )  # fmt: skip
    ax.axhline(RULE_PASS_SHARE, color=style.COLORS["up"], lw=style.LINE_WIDTH_THIN)
    _selected(ax, int(fit.rank))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.5, max(len(share), 4) + 0.5)
    ax.set_ylabel("starts passing the screen")


def _tc2c(ax, fit):
    errors = np.asarray(fit.details["elbow_errors"], float)
    ranks = np.arange(1, len(errors) + 1)
    ax.plot(ranks, errors, color=COLOR, lw=style.LINE_WIDTH, marker="o", ms=2)
    _selected(ax, int(fit.rank))
    ax.set_ylabel("normalised error")


def _stac(ax, fit):
    palette = style.CATEGORICAL
    for mode, props, rank, color in zip(
        STAC_MODES, fit.details["top_proportions"], fit.details["core_shape"],
        palette[: len(STAC_MODES)], strict=True,
    ):  # fmt: skip
        props = np.asarray(props, float)
        ax.plot(
            np.arange(1, len(props) + 1), props, color=color, lw=style.LINE_WIDTH,
            label=f"{mode} ({rank})",
        )  # fmt: skip
    ax.axhline(
        STACCATO_VAREXP / 100, color=style.COLORS["up"], lw=style.LINE_WIDTH_THIN
    )
    ax.set_yscale("log")
    ax.set_ylabel("eigenvalue share")
    ax.legend(fontsize=style.FONT_SIZE_DENSE - 1, loc="upper right", ncol=2)


DRAW = {"Pseudobulk CPD": _pb, "Tensor-cell2cell": _tc2c, "STACCato": _stac}


def _row(fig, dataset, suffix, label):
    fits, info = _fits(dataset, suffix)
    axes = fig.subplots(1, len(METHODS))
    types = f", {info['tensor_shape'][1]} types" if info else ""
    fig.suptitle(f"{label}{types}", fontsize=style.FONT_SIZE_TITLE, fontweight="normal")
    for ax, method in zip(axes, METHODS, strict=True):
        ax.grid(False)
        if fits is None:
            ax.set_axis_off()
            continue
        DRAW[method](ax, fits[method])
        ax.set_title(
            f"{method}: {fits[method].rank}", fontsize=style.FONT_SIZE_DENSE + 1
        )
        ax.set_xlabel("rank" if method != "STACCato" else "component")


def _panel(dataset, suffix, label):
    # one level down, so the row's title sits under the panel letter
    return SimpleNamespace(
        plot=lambda fig: _row(fig.subfigures(1, 1), dataset, suffix, label)
    )


PANELS = {f"{d}{s}": _panel(d, s, label) for d, s, label in ROWS}
LAYOUT = [[name] for name in PANELS]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX), LAYOUT, PANELS
    )
