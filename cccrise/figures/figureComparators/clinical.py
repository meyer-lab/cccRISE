"""Clinical association: per fit and outcome, the component with the smallest
maxT-adjusted patient-permutation p (`run_benchmark_anova`; read-depth
corrected loadings), labeled by component; circled where p < 0.05."""

import numpy as np
import pandas as pd

from .. import common, style
from . import _common

AXES = ("rect",)
OUTCOMES = {
    "6-Month Outcome (4-Way)": "6-mo. outcome",
    "6-Month CLAD/Died vs. No CLAD": "6-mo. CLAD/died",
    "1-Year CLAD/Died vs. No CLAD": "1-yr CLAD/died",
}
ALPHA = 0.05


def best() -> pd.DataFrame:
    frames = []
    for annotation in _common.ANNOTATIONS:
        table = _common.validation("benchmark_anova", annotation)
        table = table[table["RankMode"].isin(["own", "published"])
                      & table["Association"].isin(OUTCOMES)]  # fmt: skip
        if annotation == "fine":
            table = table[table["Method"] != "cccRISE"]
        table = table.assign(fit=[
            "cccRISE" if m == "cccRISE" else _common.fit_label(m, annotation) for m in table["Method"]
        ])  # fmt: skip
        frames.append(table.loc[table.groupby(["fit", "Association"])["PermPAdj"].idxmin()])
    return pd.concat(frames)


def plot(axs):
    (ax,) = axs
    table = best()
    fits = ["cccRISE"] + [_common.fit_label(m, a) for m, a in _common.FITS]
    vmax = 2.0
    for row in table.itertuples():
        x, y = fits.index(row.fit), list(OUTCOMES).index(row.Association)
        score = -np.log10(row.PermPAdj)
        ax.scatter(x, y, s=60, c=[score], cmap=style.CMAPS["weight"], vmin=0, vmax=vmax,
                   edgecolors=style.COLORS["ink"] if row.PermPAdj < ALPHA else "none",
                   linewidths=1)  # fmt: skip
        ax.annotate(f"{row.Component}", (x, y), ha="center", va="center",
                    fontsize=style.FONT_SIZE_DENSE - 1,
                    color="white" if score > 0.6 * vmax else style.COLORS["ink"])  # fmt: skip
    mappable = ax.collections[0]
    ax.set_xticks(range(len(fits)), fits, rotation=45, ha="right")
    ax.set_yticks(range(len(OUTCOMES)), list(OUTCOMES.values()))
    ax.set_xlim(-0.6, len(fits) - 0.4)
    ax.set_ylim(len(OUTCOMES) - 0.5, -0.5)
    common.colorbar(ax, mappable, "$-\\log_{10}$ maxT p")


def makeFigure():
    return common.standalone_fixed(__name__)
