"""What each CP component represents under each communication score: one row
per default-score component, with the union of every method's top LR pairs
as a dot plot, and sender and receiver cell-type heatmaps. Loadings are
divided by their largest magnitude within each component and method,
keeping sign, so methods compare by pattern rather than scale."""

import numpy as np
import pandas as pd

from ... import common, style
from . import _common as S

#: Top LR pairs per method; the dot plot shows their union.
TOP_N_LR = 3
#: Marker area (pt^2) of a dot with normalized loading magnitude 1.
DOT_AREA = 40
METHOD_TICKS = [m.removeprefix("expression_") for m in S.ALL_SCORES]
KINDS = ("lr", "sender", "receiver")


def _normalized(values: pd.Series) -> pd.Series:
    return values / values.abs().max()


def component_tables() -> list[dict[str, pd.DataFrame]]:
    """Per default component, (entry x method) normalized LR, sender, and
    receiver loadings, each method's column from its matched component."""
    columns = {}
    for method in S.ALL_SCORES:
        factors, lr_pairs = S.cp_factors(method)
        sender, receiver = S.celltype_loadings(method)
        matched = [j for _, j in S.matched_pairs(method)]
        columns[method] = {
            "lr": [_normalized(pd.Series(factors[3][:, j], lr_pairs)) for j in matched],
            "sender": [_normalized(sender.iloc[:, j]) for j in matched],
            "receiver": [_normalized(receiver.iloc[:, j]) for j in matched],
        }
    n_components = len(columns[S.DEFAULT_SCORE]["lr"])
    return [
        {
            kind: pd.DataFrame({m: columns[m][kind][i] for m in S.ALL_SCORES})
            for kind in KINDS
        }
        for i in range(n_components)
    ]


def _draw_lr_dots(ax, lr: pd.DataFrame):
    in_top = (lr.abs().rank(ascending=False, method="first") <= TOP_N_LR).any(axis=1)
    table = lr[in_top]
    table = table.loc[table.abs().mean(axis=1).sort_values(ascending=False).index]
    x, y = np.meshgrid(np.arange(table.shape[1]), np.arange(table.shape[0]))
    points = ax.scatter(
        x.ravel(),
        y.ravel(),
        s=DOT_AREA * table.abs().to_numpy().ravel(),
        c=table.to_numpy().ravel(),
        cmap=style.CMAPS["diverging"],
        vmin=-1,
        vmax=1,
    )
    ax.set_ylim(len(table) - 0.5, -0.5)
    _finish(ax, table)
    return points


def _draw_heatmap(ax, table: pd.DataFrame) -> None:
    ax.imshow(
        table.to_numpy(), cmap=style.CMAPS["diverging"], vmin=-1, vmax=1, aspect="auto"
    )
    _finish(ax, table)


def _finish(ax, table: pd.DataFrame) -> None:
    ax.set_yticks(range(len(table)), table.index, fontsize=style.FONT_SIZE_DENSE)
    ax.set_xticks(range(table.shape[1]), METHOD_TICKS, rotation=30, ha="right")
    ax.set_xlim(-0.5, table.shape[1] - 0.5)


def plot(fig):
    tables = component_tables()
    axs = fig.subplots(len(tables), 3, squeeze=False, sharex=True)
    for i, (row, component) in enumerate(zip(axs, tables, strict=True), start=1):
        points = _draw_lr_dots(row[0], component["lr"])
        _draw_heatmap(row[1], component["sender"])
        _draw_heatmap(row[2], component["receiver"])
        # Same cell types as the sender heatmap, so only that one is labeled
        row[2].sharey(row[1])
        row[2].tick_params(axis="y", labelleft=False)
        row[0].set_ylabel(f"Component {i}", fontweight="bold")
    # Method labels on the bottom row only, column titles on the top only.
    for ax in axs[:-1].ravel():
        ax.tick_params(axis="x", labelbottom=False)
    for ax, kind in zip(axs[0], ("LR pairs", "Sender", "Receiver"), strict=True):
        ax.set_title(kind)
    fig.colorbar(
        points,
        ax=axs,
        location="bottom",
        shrink=0.3,
        aspect=40,
        pad=0.04,
        label="Loading / max |loading|",
    )


def makeFigure():
    return common.standalone(plot, style.FIGSIZE["full_page"])
