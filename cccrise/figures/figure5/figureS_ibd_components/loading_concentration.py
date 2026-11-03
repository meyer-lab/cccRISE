"""How many biopsies carry each CP component: the cumulative share of its
summed read-depth-corrected loading over biopsies, largest first (the
dashed line marks the top `TOP`). The legend gives each component's
Kruskal-Wallis p across inflammation statuses once its top `TOP` biopsies
are left out: C1's and C5's associations do not rest on their few heavy
biopsies, C6's does."""

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from ... import common, style
from .. import _common as S

AXES = ("rect",)
TOP = 5


def plot(axs):
    (ax,) = axs
    loadings = S.corrected_loadings("cp_A")
    status = S.samples()[S.GROUP_COLUMN].to_numpy()
    rows = []
    for k in loadings.columns:
        v = loadings[k].clip(lower=0).to_numpy()
        order = np.argsort(-v)
        share = np.cumsum(v[order]) / v.sum()
        rest = order[TOP:]
        p = kruskal(*[v[rest][status[rest] == g] for g in S.GROUP_LABELS]).pvalue
        ax.plot(
            np.arange(1, len(v) + 1),
            share,
            color=style.CATEGORICAL[k - 1],
            label=f"C{k} ({p:.0e})",
        )
        rows += [
            {
                "Component": k,
                "Biopsies": n + 1,
                "Cumulative share": s,
                f"Status p without top {TOP}": p,
            }
            for n, s in enumerate(share)
        ]
    common.source_data(
        "loading_concentration",
        pd.DataFrame(rows),
        f"Cumulative share of each component's summed read-depth-corrected loading "
        f"(negative loadings as 0) over biopsies, largest first, and the "
        f"Kruskal-Wallis p across statuses without its top {TOP} biopsies.",
    )
    common.reference_line(ax, TOP, horizontal=False)
    ax.set_xscale("log")
    ax.set_xlabel("Biopsies (largest loading first)")
    ax.set_ylabel("Cumulative share\nof loading")
    common.legend(
        ax, "right", fontsize=style.FONT_SIZE_DENSE, title=f"Status p,\nno top {TOP}"
    )


def makeFigure():
    return common.standalone_fixed(__name__)
