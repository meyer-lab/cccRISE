"""Where each CP comparator component sits: the effective number of samples
of its condition factor against the effective number of cell types of its
sender and receiver factors (participation ratios). Most are one cell type,
and many are one sample. STACCato is not drawn (a Tucker fit's condition,
sender and receiver columns are not paired into components), nor is cccRISE
(its per-cell loadings have no one-number-per-type factor to compare)."""

import matplotlib.ticker as mticker
import numpy as np

from .. import common, style
from ._common import SHORT, load

CP_METHODS = ("Pseudobulk CPD", "Tensor-cell2cell")

MARKERS = {"coarse": "o", "fine": "D"}


def plot(fig):
    table = load()
    table = table[table["method"].isin(CP_METHODS)]
    wide = table.pivot_table(
        index=["method", "granularity", "component"],
        columns=["mode", "statistic"],
        values="value",
    )
    types = wide[[("sender", "type_breadth"), ("receiver", "type_breadth")]].max(axis=1)
    samples = wide[("condition", "sample_breadth")]
    ax = fig.subplots()
    colors = dict(zip(CP_METHODS, style.CATEGORICAL[1:], strict=False))
    for (method, granularity), idx in samples.groupby(level=[0, 1]).groups.items():
        ok = types.loc[idx].notna()
        ax.scatter(
            samples.loc[idx][ok],
            types.loc[idx][ok],
            marker=MARKERS[granularity],
            s=16,
            color=colors[method],
            facecolor=colors[method] if granularity == "coarse" else "none",
            label=f"{SHORT.get(method, method)} {granularity}",
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks([1, 2, 3, 5, 8])
    ax.axvline(1.5, color=style.COLORS["reference"], ls=":", lw=0.8)
    ax.axhline(1.5, color=style.COLORS["reference"], ls=":", lw=0.8)
    single = ((samples < 1.5) & (types < 1.5)).sum()
    ax.text(
        0.03,
        0.97,
        f"{single} of {int(types.notna().sum())} components: one sample, one type",
        transform=ax.transAxes,
        fontsize=style.FONT_SIZE_DENSE,
        va="top",
    )
    ax.set_xlabel("effective no. of samples (condition factor)")
    ax.set_ylabel("effective no. of cell types\n(max of sender, receiver)")
    ax.set_ylim(0.9, np.nanmax(types) + 0.3)
    ax.legend(fontsize=style.FONT_SIZE_DENSE - 1, frameon=False, loc="center right")
    ax.grid(False)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
