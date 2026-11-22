"""Fraction of each ALAD sample's cells in each annotated cell type by ALAD
status (control, recovered, declined), with each cell type's Kruskal-Wallis q
(Benjamini-Hochberg)."""

from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
from scipy.stats import kruskal

from ....anova import adjust_pvalues, get_sample_to_group
from ....runners.datasets import DATASETS
from ... import common, style

DATASET = "alad"
# Drawn in this order; each status keeps its color
STATUS_COLORS = {
    "control": style.CATEGORICAL[0],
    "recovered": style.CATEGORICAL[2],
    "declined": style.CATEGORICAL[1],
}
Y_TICKS = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
AXES = ("half",)
Y_TOP = 1.3  # room above a fraction of 1 for the q values
Q_Y = 0.98  # axes fraction: the q values, just under the top
KEY_TOP = 0.9  # axes fraction: the legend, under the q values


def proportions() -> pd.DataFrame:
    """Fraction of each sample's (rows) cells in each cell type (columns, in
    `style.CELL_TYPES` order), with the sample's ALAD "Status"."""
    spec = DATASETS[DATASET]
    adata = spec.load()
    obs = adata.obs
    fractions = pd.crosstab(
        obs[spec.condition_column], obs[spec.celltype_column], normalize="index"
    ).reindex(columns=style.CELL_TYPES[DATASET], fill_value=0)
    assert spec.group_column is not None
    status = get_sample_to_group(adata, spec.condition_column, spec.group_column)
    return fractions.assign(Status=status.reindex(fractions.index).to_numpy())


def plot(axs):
    (ax,) = axs
    df = proportions()
    celltypes = style.CELL_TYPES[DATASET]
    q = composition_q(df, celltypes)
    counts = df["Status"].value_counts()
    common.source_data(
        "celltype_fraction_per_sample",
        df.rename_axis("Sample")
        .reset_index()
        .melt(
            id_vars=["Sample", "Status"], var_name="Cell type", value_name="Fraction"
        ),
        "Fraction of each BAL sample's cells annotated as each broad cell type, "
        "with the sample's ALAD status.",
    )
    common.source_data(
        "celltype_fraction_kruskal",
        pd.DataFrame({"Cell type": celltypes, "Kruskal-Wallis q": q}),
        "Kruskal-Wallis test of each cell type's per-sample fraction across the "
        "three ALAD statuses; p-values Benjamini-Hochberg corrected across cell "
        "types (q).",
    )
    labels = {s: f"{s.capitalize()} (n={counts[s]})" for s in STATUS_COLORS}
    draw(ax, df, celltypes, q, STATUS_COLORS, labels)


def draw(
    ax,
    df: pd.DataFrame,
    celltypes: list[str],
    q: np.ndarray,
    colors: dict[str, str],
    labels: dict[str, str],
    vertical: bool = False,
) -> None:
    """Per-sample fraction of each of `celltypes` (`df` columns) by "Status"
    (`colors`' keys, in order; legend `labels`), square-root scaled, each
    cell type's `q` (inside the axes' top, the legend under them). `vertical` turns the cell type names and the q
    values on their side, for many cell types."""
    shared: dict[str, Any] = {
        "data": df.melt(id_vars="Status", var_name="Cell type", value_name="Fraction"),
        "x": "Cell type",
        "y": "Fraction",
        "order": celltypes,
        "hue": "Status",
        "hue_order": list(colors),
        "dodge": True,
        "ax": ax,
    }
    # stripplot dodges its points over 0.8 of each category; boxes dodged over
    # the same width line up with them, `gap` keeping each box 0.7 / n wide
    sns.boxplot(
        **shared,
        palette=colors,
        width=0.8,
        gap=0.125,
        fliersize=0,
        linewidth=style.LINE_WIDTH_THIN,
        boxprops={"alpha": 0.8},
    )
    common.stripplot(
        **shared,
        palette=[style.COLORS["ink"]] * len(colors),
        size=0.8 if vertical else 1.5,
        jitter=0.08,
        legend=False,
    )
    # Square-root scale, so the minor cell types are not flattened by the major
    ax.set_yscale("function", functions=(np.sqrt, np.square))
    ax.set_yticks(Y_TICKS, [f"{t:g}" for t in Y_TICKS])
    ax.set_ylim(0, Y_TOP)
    _label_celltypes(ax, celltypes, q, vertical)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=[labels[s] for s in colors],
        loc="upper right",
        bbox_to_anchor=(1, KEY_TOP),
    )


def _label_celltypes(ax, celltypes: list[str], q: np.ndarray, vertical: bool) -> None:
    """Name each cell type below its boxes and write its q just under the
    axes' top."""
    if vertical:
        ax.set_xticks(
            range(len(celltypes)),
            celltypes,
            rotation=90,
            fontsize=style.FONT_SIZE_DENSE,
        )
    else:
        ax.set_xticks(
            range(len(celltypes)),
            celltypes,
            rotation=40,
            ha="right",
            rotation_mode="anchor",
        )
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of cells")
    top = blended_transform_factory(ax.transData, ax.transAxes)
    for x, value in enumerate(q):
        ax.text(
            x,
            Q_Y,
            f"{value:.2g}",
            transform=top,
            ha="center",
            va="top",
            rotation=90 if vertical else 0,
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold" if vertical and value < common.SIG_ALPHA else "normal",
        )


def composition_q(
    df: pd.DataFrame, celltypes: list[str], statuses=tuple(STATUS_COLORS)
) -> np.ndarray:
    """Each cell type's Kruskal-Wallis p across `statuses`, Benjamini-Hochberg
    corrected across `celltypes`."""
    return adjust_pvalues(
        [
            kruskal(*(df.loc[df["Status"] == s, c] for s in statuses)).pvalue
            for c in celltypes
        ]
    )


def makeFigure():
    return common.standalone_fixed(__name__)
