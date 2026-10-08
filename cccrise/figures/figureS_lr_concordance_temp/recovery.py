"""Which cccRISE components the comparators recover (draft panel A).

Per cccRISE component and comparator fit (own rank; coarse and fine
annotation), the fit's column that best matches the component's top-10 LR
pairs. A match is scored by the pairs both top-10s share: the geometric mean
of the share of the component's squared loading and the share of the
column's squared loading they carry, sqrt(m_cccRISE x m_comparator). It is 1
only when both put all their weight on the shared pairs, and 0 when either
puts none there; by Cauchy-Schwarz it bounds what the shared pairs add to the
|cosine| of the two LR columns. Colour: that score for the best column (ties:
more shared pairs). Number: how many pairs it shares."""

import numpy as np
import pandas as pd

from .. import common, style
from ..figureS_lr_concordance._common import ANNOTATIONS, METHODS, SHORT, load

K = 10
FITS = [(a, m) for a in ANNOTATIONS for m in METHODS]


def best(table: pd.DataFrame) -> pd.DataFrame:
    """Per (cccRISE component, annotation, method): the column with the
    largest joint loading on the shared top-`K` pairs (`joint`), then the
    most shared pairs."""
    table = table[(table["k"] == K) & table["method"].isin(METHODS)]
    table = table.assign(
        joint=np.sqrt(table["mass_cccrise"] * table["mass_comparator"])
    )
    table = table.sort_values(["joint", "shared"], ascending=False, kind="stable")
    keys = ["cccrise_component", "annotation", "method"]
    return table.drop_duplicates(keys).set_index(keys).sort_index()


def _grid(rows, comps, name):
    return np.array([[rows.loc[(c, *fit), name] for fit in FITS] for c in comps])


def _annotate(ax, shared, joint):
    for (i, j), n in np.ndenumerate(shared):
        ax.text(
            j, i, f"{int(n)}", ha="center", va="center",
            color="white" if joint[i, j] > 0.6 else style.COLORS["ink"],
            fontsize=style.FONT_SIZE_DENSE + 1, fontweight="bold",
        )  # fmt: skip


def plot(fig):
    rows = best(load("overlap"))
    comps = sorted(rows.index.get_level_values(0).unique())
    joint = _grid(rows, comps, "joint")
    ax = fig.subplots()
    image = ax.imshow(joint, cmap=style.CMAPS["weight"], vmin=0, vmax=1, aspect="auto")
    _annotate(ax, _grid(rows, comps, "shared"), joint)
    ax.set_xticks(range(len(FITS)), [f"{SHORT[m]}\n{a}" for a, m in FITS])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0)
    ax.set_yticks(range(len(comps)), [f"C{c}" for c in comps])
    ax.axvline(len(METHODS) - 0.5, color="white", lw=3)
    ax.set_ylabel("cccRISE component")
    ax.grid(False)
    fig.colorbar(
        image,
        ax=ax,
        label="joint loading on shared pairs\n√(cccRISE share × comparator share)",
        shrink=0.8,
        pad=0.01,
    )


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 55 * style.MM))
