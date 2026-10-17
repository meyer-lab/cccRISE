"""Annotated cell type proportion of each ALAD sample by ALAD status (control,
declined, recovered), with each cell type's Kruskal-Wallis q across the three
statuses, Benjamini-Hochberg corrected across cell types."""

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
STATUS_COLORS = dict(
    zip(("control", "declined", "recovered"), style.CATEGORICAL, strict=False)
)
Y_TICKS = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
AXES = ("full",)


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
    long = df.melt(id_vars="Status", var_name="Cell type", value_name="Fraction")
    counts = df["Status"].value_counts()
    shared: dict[str, Any] = {
        "data": long,
        "x": "Cell type",
        "y": "Fraction",
        "order": celltypes,
        "hue": "Status",
        "hue_order": list(STATUS_COLORS),
        "dodge": True,
        "ax": ax,
    }
    sns.boxplot(
        **shared,
        palette=STATUS_COLORS,
        width=0.7,
        fliersize=0,
        linewidth=style.LINE_WIDTH_THIN,
        boxprops={"alpha": 0.8},
    )
    sns.stripplot(
        **shared,
        palette=[style.COLORS["ink"]] * len(STATUS_COLORS),
        size=1.5,
        jitter=0.08,
        legend=False,
    )
    # Square-root scale, so the minor cell types are not flattened by macrophages
    ax.set_yscale("function", functions=(np.sqrt, np.square))
    ax.set_yticks(Y_TICKS, [f"{t:g}" for t in Y_TICKS])
    ax.set_ylim(0, 1)
    ax.set_xticks(
        range(len(celltypes)), ["\n".join(c.rsplit(" ", 1)) for c in celltypes]
    )
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of cells")
    top = blended_transform_factory(ax.transData, ax.transAxes)
    for x, value in enumerate(q):
        ax.text(
            x,
            1,
            f"q = {value:.2f}",
            transform=top,
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
        )
    handles, _ = ax.get_legend_handles_labels()
    common.legend(
        ax,
        "above",
        handles=handles,
        labels=[f"{s.capitalize()} (n={counts[s]})" for s in STATUS_COLORS],
        ncols=len(STATUS_COLORS),
    )


def composition_q(df: pd.DataFrame, celltypes: list[str]) -> np.ndarray:
    """Each cell type's Kruskal-Wallis p across `STATUS_COLORS`' statuses,
    Benjamini-Hochberg corrected across `celltypes`."""
    return adjust_pvalues(
        [
            kruskal(*(df.loc[df["Status"] == s, c] for s in STATUS_COLORS)).pvalue
            for c in celltypes
        ]
    )


def makeFigure():
    return common.standalone_fixed(__name__)
