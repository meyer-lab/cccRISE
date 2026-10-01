"""
Figure: what each CP component represents under each communication-scoring
method on ALAD.

One row per CP component (the default score's, core-fit numbering), with
three panels: the union of every method's top ligand-receptor pairs as a dot
plot, and sender and receiver cell-type heatmaps. Each alternate method's
component is the one `figureScoringComparison.match_components` pairs with the
default's. Loadings are divided by their largest magnitude within each
component and method, keeping sign, so methods compare by pattern rather than
scale.

Only loads results, from ``VALIDATION_DIR/alt_score_methods/<method>/`` (generated
by `cccrise-altcommscore`).
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .common import getSetup, subplotLabel
from .figureScoringComparison import (
    ALL_SCORES,
    DEFAULT_SCORE,
    SPEC,
    celltype_loadings,
    cp_factors,
    match_components,
)

#: Top LR pairs per method; the dot plot shows the union over methods.
TOP_N_LR = 5
CMAP = "RdBu_r"
#: Marker area of a dot with normalized loading magnitude 1.
DOT_AREA = 120
METHOD_TICKS = [m.removeprefix("expression_") for m in ALL_SCORES]


def _normalized(values: pd.Series) -> pd.Series:
    return values / values.abs().max()


def _component_tables(adata_raw) -> list[dict[str, pd.DataFrame]]:
    """Per default-score component, (entry x method) tables of normalized LR,
    sender and receiver loadings, each method's column taken from its matched
    component."""
    columns: dict[str, dict[str, list[pd.Series]]] = {}
    for method in ALL_SCORES:
        factors, lr_symbols = cp_factors(method)
        sender, receiver = celltype_loadings(method, adata_raw)
        matched = match_components(DEFAULT_SCORE, method)["MatchedComponent"] - 1
        columns[method] = {
            "lr": [
                _normalized(pd.Series(factors[3][:, j], lr_symbols)) for j in matched
            ],
            "sender": [_normalized(sender.iloc[:, j]) for j in matched],
            "receiver": [_normalized(receiver.iloc[:, j]) for j in matched],
        }
    n_components = len(columns[DEFAULT_SCORE]["lr"])
    return [
        {
            kind: pd.DataFrame({m: columns[m][kind][i] for m in ALL_SCORES})
            for kind in ("lr", "sender", "receiver")
        }
        for i in range(n_components)
    ]


def _top_mask(table: pd.DataFrame, n: int) -> pd.DataFrame:
    """True where an entry is among its column's `n` largest magnitudes."""
    return table.abs().rank(ascending=False, method="first") <= n


def _draw_lr_dots(ax: plt.Axes, lr: pd.DataFrame, title: str):
    union = _top_mask(lr, TOP_N_LR).any(axis=1)
    table = lr[union].loc[
        lambda t: t.abs().mean(axis=1).sort_values(ascending=False).index
    ]
    x, y = np.meshgrid(np.arange(table.shape[1]), np.arange(table.shape[0]))
    points = ax.scatter(
        x.ravel(),
        y.ravel(),
        s=DOT_AREA * table.abs().to_numpy().ravel(),
        c=table.to_numpy().ravel(),
        cmap=CMAP,
        vmin=-1,
        vmax=1,
    )
    ax.set_yticks(range(len(table)), table.index, fontsize=5)
    ax.set_ylim(len(table) - 0.5, -0.5)
    _finish(ax, table, title)
    return points


def _draw_celltype_heatmap(ax: plt.Axes, table: pd.DataFrame, title: str) -> None:
    ax.imshow(table.to_numpy(), cmap=CMAP, vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(table)), table.index, fontsize=5)
    _finish(ax, table, title)


def _finish(ax: plt.Axes, table: pd.DataFrame, title: str) -> None:
    ax.set_xticks(range(table.shape[1]), METHOD_TICKS, fontsize=6)
    ax.set_xlim(-0.5, table.shape[1] - 0.5)
    ax.grid(False)
    ax.set_title(title, fontsize=8)


def makeFigure():
    tables = _component_tables(SPEC.load())
    axs, fig = getSetup((10, 3 * len(tables)), (len(tables), 3))
    for i, component in enumerate(tables):
        points = _draw_lr_dots(axs[3 * i], component["lr"], f"Comp. {i + 1}: LR pairs")
        for j, kind in enumerate(("sender", "receiver"), start=1):
            _draw_celltype_heatmap(
                axs[3 * i + j], component[kind], f"Comp. {i + 1}: {kind} cell types"
            )
    fig.colorbar(
        points,
        ax=axs,
        shrink=0.25,
        label="Loading / max |loading|",
    )
    subplotLabel(axs)
    return fig
