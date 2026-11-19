"""Each default-score CPD component's paired component under each communication
score: its top LR pair loadings and its mean sender and receiver weight per
cell type, each divided by its largest absolute value.

What each CPD component represents under each communication score: one row
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
METHOD_TICKS = [
    S.SCORE_LABELS[m].replace("Geometric mean", "Gmean") for m in S.ALL_SCORES
]
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
    table = _shown_pairs(lr)
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
    ax.set_xticks(range(table.shape[1]), METHOD_TICKS, fontsize=style.FONT_SIZE_DENSE)
    ax.set_xlim(-0.5, table.shape[1] - 0.5)


def save_source_data(tables: list[dict[str, pd.DataFrame]]) -> None:
    """The plotted normalized loadings per default component, and the per-cell
    weights the cell type heatmaps average."""
    labels = {m: S.SCORE_LABELS[m] for m in S.ALL_SCORES}

    def long(kind: str, entry: str, select) -> pd.DataFrame:
        return pd.concat(
            select(table[kind])
            .rename(columns=labels)
            .rename_axis(entry)
            .reset_index()
            .melt(id_vars=entry, var_name="Method", value_name="Normalized loading")
            .assign(Component=i)
            for i, table in enumerate(tables, start=1)
        )

    common.source_data(
        "loading_lr_by_score",
        long("lr", "LR pair", _shown_pairs),
        "One row per (default-score CPD component, LR pair shown, score): the LR "
        "loading of the score's paired component divided by its largest "
        f"|loading| (dot color and area); shown are each score's top {TOP_N_LR} "
        "pairs.",
        part=0,
    )
    common.source_data(
        "loading_celltype_by_score",
        pd.concat(
            long(kind, "Cell type", lambda t: t).assign(Role=role)
            for kind, role in (("sender", "Sender"), ("receiver", "Receiver"))
        ),
        "One row per (default-score CPD component, role, cell type, score): the "
        "cell type's mean per-cell sender or receiver weight on the score's "
        "paired component, divided by its largest |mean| across cell types "
        "(the heatmaps).",
        part=1,
    )
    common.source_data(
        "weight_cells_by_score",
        cell_weights(),
        "One row per cell: its cell type and, for each score, its per-cell "
        "sender and receiver weight on the component paired with each "
        "default-score component (numbered by the default score), the values "
        "the cell type heatmaps average.",
        part=1,
    )


def cell_weights() -> pd.DataFrame:
    """Per cell, every score's sender and receiver weight per default
    component."""
    ad = S.method_adata(S.DEFAULT_SCORE)
    columns = {
        "Cell": ad.obs_names,
        "Cell type": ad.obs[S.SPEC.celltype_column].to_numpy(),
    }
    for method in S.ALL_SCORES:
        obsm = S.method_adata(method).obsm
        for i, j in S.matched_pairs(method):
            for key, role in (("sc_B", "sender"), ("rc_C", "receiver")):
                columns[f"{S.SCORE_LABELS[method]} {role} cmp {i + 1}"] = obsm[key][
                    :, j
                ]
    return pd.DataFrame(columns)


def _shown_pairs(lr: pd.DataFrame) -> pd.DataFrame:
    """The LR pairs `_draw_lr_dots` shows: each method's top `TOP_N_LR`."""
    return lr[(lr.abs().rank(ascending=False, method="first") <= TOP_N_LR).any(axis=1)]


def plot(fig):
    tables = component_tables()
    save_source_data(tables)
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
        ax=axs[:, 1:],
        location="right",
        aspect=60,
        label="Loading / max |loading|",
    )


def makeFigure():
    return common.standalone(plot, style.FIGSIZE["full_page"])
