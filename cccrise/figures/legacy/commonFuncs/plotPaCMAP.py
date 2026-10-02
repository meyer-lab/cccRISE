from pathlib import Path

import anndata
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ...common import getSetup, subplotLabel


def _get_canvas(points: np.ndarray) -> ds.Canvas:
    """Compute bounds on a space with appropriate padding.

    Not rounded to integers: PaCMAP coordinates span enough range that
    rounding barely mattered (axis ticks are hidden anyway, see
    `assign_labels`), but other 2D spaces plotted the same way -- e.g.
    sender/receiver weight, spanning a fraction of a unit -- would have
    both bounds round to the same integer and hand datashader a
    zero-width canvas.
    """
    min_xy = np.nanmin(points, axis=0)
    assert min_xy.size == 2
    max_xy = np.nanmax(points, axis=0)

    mins = min_xy - 0.05 * (max_xy - min_xy)
    maxs = max_xy + 0.05 * (max_xy - min_xy)

    canvas = ds.Canvas(
        plot_width=300,
        plot_height=300,
        x_range=(mins[0], maxs[0]),
        y_range=(mins[1], maxs[1]),
    )

    return canvas


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


def ds_show(result, ax, canvas: ds.Canvas):
    """Draw a shaded datashader aggregation, mapped onto `canvas`'s actual
    data coordinates.

    `imshow` defaults to plain pixel coordinates (0..plot_width,
    0..plot_height) when given no `extent`, leaving the axes disconnected
    from the real x/y values -- harmless for a plot nothing else is drawn
    on (e.g. a PaCMAP embedding, whose orientation carries no meaning), but
    wrong for anything overlaid afterward in data coordinates (e.g. a
    threshold line), which ends up pinned near an axis edge instead of at
    its actual value.
    """
    result = tf.set_background(result, "white")
    assert canvas.x_range is not None
    assert canvas.y_range is not None
    x0, x1 = canvas.x_range
    y0, y1 = canvas.y_range
    ax.imshow(result.to_pil(), extent=(x0, x1, y0, y1), origin="upper")


def _plot_shaded_pacmap(
    points,
    values,
    ax: Axes,
    cmap,
    span: tuple[float, float],
    title: str,
    cbar_match_height: bool = False,
):
    """Shade PaCMAP points by a continuous value and draw the result with a
    colorbar. `cbar_match_height` sizes the colorbar to the (equal-aspect)
    image rather than the full axes slot."""
    canvas = _get_canvas(points)
    data = pd.DataFrame(points, columns=pd.Index(["x", "y"]))
    data["val_cat"] = values
    result = tf.shade(
        agg=canvas.points(data, "x", "y", agg=ds.mean("val_cat")),
        cmap=cmap,
        span=span,
        how="linear",
        min_alpha=255,
    )

    ds_show(result, ax, canvas)

    sm = ScalarMappable(norm=matplotlib.colors.Normalize(*span), cmap=cmap)
    if cbar_match_height:
        # An inset follows the drawn (aspect-adjusted) axes box, and stays out
        # of `fig.axes` for callers that expect every axes to sit on a grid.
        plt.colorbar(sm, cax=ax.inset_axes((1.03, 0.0, 0.04, 1.0)))
    else:
        plt.colorbar(sm, ax=ax)
    ax.set(title=title)
    assign_labels(ax)


def plot_wc_pacmap(
    X: anndata.AnnData,
    cmp: int,
    ax: Axes,
    cbarMax: float = 1.0,
    factor_matrix: str | None = None,
    cbar_match_height: bool = False,
):
    """Scatterplot of PaCMAP visualization of weighted cells for one component in receiver or sender"""
    if factor_matrix == "B":
        factor = "sc_B"
    elif factor_matrix == "C":
        factor = "rc_C"
    sc_factor = X.obsm[factor]
    values = sc_factor[:, cmp - 1]
    points = np.array(X.obsm["PaCMAP"])

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    values = values / np.max(values)

    _plot_shaded_pacmap(
        points,
        values,
        ax,
        cmap,
        (-cbarMax, cbarMax),
        "Cmp. " + str(cmp),
        cbar_match_height=cbar_match_height,
    )


def plot_eigenstate_pacmap(
    X: anndata.AnnData,
    eigenstate: int,
    ax: Axes,
    span_percentile: float = 99,
    cbar_match_height: bool = False,
):
    """PaCMAP shaded by each cell's projection onto one RISE eigenstate
    (1-indexed). Projections are heavy-tailed, so the symmetric color span is
    the `span_percentile` of |projection| rather than its max; cells beyond it
    saturate."""
    values = np.asarray(X.obsm["projections"])[:, eigenstate - 1]
    limit = float(np.percentile(np.abs(values), span_percentile))
    _plot_shaded_pacmap(
        np.asarray(X.obsm["PaCMAP"]),
        values,
        ax,
        sns.diverging_palette(240, 10, as_cmap=True),
        (-limit, limit),
        f"Eigenstate {eigenstate} Weight",
        cbar_match_height=cbar_match_height,
    )


def make_wc_pacmap_figure(
    data_or_path: str | Path | anndata.AnnData,
    grid: tuple[int, int],
    figsize: tuple[int, int] = (18, 18),
) -> Figure:
    """Build a grid figure of weighted sender/receiver cell PaCMAPs for every component."""
    from ...common import get_alad_data, get_covid_data

    ax, f = getSetup(figsize, grid)
    subplotLabel(ax)

    if isinstance(data_or_path, anndata.AnnData):
        X = data_or_path
    elif "covid" in str(data_or_path).lower():
        X = get_covid_data(project_cells=True, do_embedding=True)
    elif "alad" in str(data_or_path).lower():
        X = get_alad_data(project_cells=True, do_embedding=True)
    else:
        X = anndata.read_h5ad(data_or_path)

    cp_rank = X.uns["A"].shape[1]

    for i in range(cp_rank):
        plot_wc_pacmap(X, i + 1, ax[i], factor_matrix="B", cbarMax=0.3)
        ax[i].set_title(f"Cmp.{i + 1} Sender Cells")

    for i in range(cp_rank):
        plot_wc_pacmap(X, i + 1, ax[i + cp_rank], factor_matrix="C", cbarMax=0.3)
        ax[i + cp_rank].set_title(f"Cmp.{i + 1} Receiver Cells")

    return f


def plot_categorical_scatter(
    points: np.ndarray,
    labels: pd.Series,
    ax: Axes,
    cmap: str = "tab20",
    color_key=None,
) -> None:
    """Datashader-aggregated scatter of arbitrary 2D points colored by a
    categorical label, with a legend.

    Point clouds here run to tens of thousands of cells; a plain matplotlib
    scatter redrawn on every Shiny interaction is both slow and prone to
    overplotting, so this always goes through datashader's aggregation
    (`ds_show`) instead. Shared by `plot_labels_pacmap` (PaCMAP coordinates)
    and any other 2D space -- e.g. sender vs. receiver weight -- that needs
    the same large-N-safe categorical rendering.
    """
    if labels.dtype == "category":
        labels = labels.cat.set_categories(
            np.sort(labels.cat.categories.values), ordered=True
        )
    indices = np.argsort(labels)

    points = points[indices, :]
    labels = labels.iloc[indices]

    canvas = _get_canvas(points)
    data = pd.DataFrame(points, columns=pd.Index(["x", "y"]))

    data["label"] = pd.Categorical(labels)
    aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))

    unique_labels = np.unique(labels)
    num_labels = unique_labels.shape[0]
    if color_key is None:
        color_key = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, num_labels)))
    legend_elements = [
        Patch(facecolor=color_key[i], label=k) for i, k in enumerate(unique_labels)
    ]
    result = tf.shade(
        aggregation,
        color_key=color_key,
        how="eq_hist",
        min_alpha=255,
    )

    ds_show(result, ax, canvas)
    ax.legend(handles=legend_elements)


def plot_labels_pacmap(
    X: anndata.AnnData,
    labelType: str,
    ax: Axes,
    condition=None,
    cmap="tab20",
    color_key=None,
):
    """Scatterplot of PaCMAP visualization weighted by condition or cell type"""
    labels = pd.Series(X.obs[labelType])

    if condition is not None:
        labels = pd.Series([c if c in condition else "Z Other" for c in labels])

    points = np.array(X.obsm["PaCMAP"])
    plot_categorical_scatter(points, labels, ax, cmap=cmap, color_key=color_key)
    assign_labels(ax)


def plot_wc_per_celltype(
    X: anndata.AnnData,
    cmp: int,
    ax: Axes,
    outliers: bool = False,
    cellType="cell_type",
    factor_matrix: str | None = None,
    color=None,
):
    """Boxplot of weighted cells for one component across cell types"""
    if factor_matrix == "B":
        factor = "sc_B"
    elif factor_matrix == "C":
        factor = "rc_C"
    sc_factor = X.obsm[factor]
    XX = sc_factor[:, cmp - 1]
    cmpName = f"Cmp. {cmp}"

    df = pd.DataFrame({cmpName: XX, "Cell Type": X.obs[cellType].to_numpy()})

    sns.boxplot(
        data=df,
        x=cmpName,
        y="Cell Type",
        showfliers=outliers,
        color=color,
        ax=ax,
    )
    maxvalue = np.max(np.abs(ax.get_xticks()))
    ax.set(xticks=np.linspace(-maxvalue, maxvalue, num=5), xlabel="WC Weight")
    ax.set_title(cmpName)


def assign_labels(ax):
    ax.set(xlabel="PaCMAP1", ylabel="PaCMAP2", xticks=[], yticks=[])
    return ax
