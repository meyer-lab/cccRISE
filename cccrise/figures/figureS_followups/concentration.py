"""How concentrated the CP comparators' own-rank components are on each
tensor: the share that sit on one cell type (effective number of sender
and receiver types < 1.5), and the share on one sample and one type."""

import numpy as np

from .. import common, style
from ._common import SHORT, VARIANT_LABELS, load

CP = ("Pseudobulk CPD", "Tensor-cell2cell")


def _shares(table):
    wide = table.pivot_table(
        index=["variant", "method", "component"],
        columns=["mode", "statistic"],
        values="value",
    )
    types = wide[[("sender", "type_breadth"), ("receiver", "type_breadth")]].max(axis=1)
    one_type = types < 1.5
    one_both = one_type & (wide[("condition", "sample_breadth")] < 1.5)
    return one_type.groupby(level=[0, 1]).mean(), one_both.groupby(level=[0, 1]).mean()


def plot(fig):
    table = load("program_breadth")
    table = table[table["method"].isin(CP)]
    one_type, one_both = _shares(table)
    ax = fig.subplots()
    x = np.arange(len(VARIANT_LABELS))
    for i, method in enumerate(CP):
        offset = (i - 0.5) * 0.38
        color = style.CATEGORICAL[i + 1]
        a = [one_type.get((v, method), np.nan) for v in VARIANT_LABELS]
        b = [one_both.get((v, method), np.nan) for v in VARIANT_LABELS]
        ax.bar(
            x + offset,
            a,
            0.36,
            color=color,
            alpha=0.35,
            label=f"{SHORT[method]}: one type",
        )
        ax.bar(
            x + offset,
            b,
            0.36,
            color=color,
            label=f"{SHORT[method]}: one sample + one type",
        )
    ax.set_xticks(x, list(VARIANT_LABELS.values()))
    ax.set_ylabel("share of components")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=style.FONT_SIZE_DENSE - 1, frameon=False, loc="upper right")
    ax.grid(False)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
