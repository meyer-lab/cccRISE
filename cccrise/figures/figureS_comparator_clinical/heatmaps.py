"""Eta-squared of every component of every method against each clinical
association, one heatmap per method (as Figure 3's association panel). Stars:
maxT permutation p < 0.05, family-wise across the method's components."""

import numpy as np
import pandas as pd

from .. import common, style
from ._common import METHODS, fit_label, load


def _heatmap(ax, table: pd.DataFrame, order: list[str], vmax: float):
    effect, perm = (
        table.pivot(index="Association", columns="Component", values=v).reindex(order)
        for v in ("EtaSquared", "PermPAdj")
    )
    values = effect.to_numpy(float)
    image = ax.imshow(
        values, cmap=style.CMAPS["sequential"], vmin=0, vmax=vmax, aspect="auto"
    )
    for (i, j), value in np.ndenumerate(values):
        significant = perm.iloc[i, j] < common.SIG_ALPHA
        ax.text(
            j,
            i,
            f"{value:.2f}" + ("*" if significant else ""),
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
            fontweight="bold" if significant else "normal",
            color="white" if value > 0.6 * vmax else style.COLORS["ink"],
        )
    ax.set_xticks(range(effect.shape[1]), effect.columns)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    return image


def plot_granularity(fig, granularity: str) -> None:
    anova = load("anova", granularity)
    if anova is None:
        common.placeholder(fig, f"{granularity} fits not run yet")
        return
    order = list(dict.fromkeys(anova["Association"]))
    fits = [
        (m, anova.loc[anova["Method"] == m, "Rank"].iloc[0])
        for m in METHODS
        if (anova["Method"] == m).any()
    ]
    widths = [int(anova.loc[anova["Method"] == m, "Component"].max()) for m, _ in fits]
    axes = fig.subplots(1, len(fits), width_ratios=widths, sharey=True)
    # One scale for both granularities, so panels compare by colour.
    vmax = max(
        float(table["EtaSquared"].max())
        for g in ("coarse", "fine")
        if (table := load("anova", g)) is not None
    )
    for ax, (method, rank) in zip(axes, fits, strict=True):
        image = _heatmap(ax, anova[anova["Method"] == method], order, vmax)
        ax.set_title(fit_label(method, rank))
    axes[0].set_yticks(range(len(order)), order, fontsize=style.FONT_SIZE_DENSE)
    fig.supxlabel("Component", fontsize=style.FONT_SIZE_DENSE)
    fig.colorbar(image, ax=axes, label=r"$\eta^2$", shrink=0.8)


def plot(fig):
    plot_granularity(fig, "coarse")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
