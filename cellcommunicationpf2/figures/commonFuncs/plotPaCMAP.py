import anndata
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch


def _get_canvas(points: np.ndarray) -> ds.Canvas:
    """Compute bounds on a space with appropriate padding"""
    min_xy = np.nanmin(points, axis=0)
    assert min_xy.size == 2
    max_xy = np.nanmax(points, axis=0)

    mins = np.round(min_xy - 0.05 * (max_xy - min_xy))
    maxs = np.round(max_xy + 0.05 * (max_xy - min_xy))

    canvas = ds.Canvas(
        plot_width=300,
        plot_height=300,
        x_range=(mins[0], maxs[0]),
        y_range=(mins[1], maxs[1]),
    )

    return canvas


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


def ds_show(result, ax):
    result = tf.set_background(result, "white")
    ax.imshow(result.to_pil())


def _plot_shaded_pacmap(
    points, values, ax: Axes, cmap, span: tuple[float, float], title: str
):
    """Shade PaCMAP points by a continuous value and draw the result with a colorbar"""
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

    ds_show(result, ax)

    psm = plt.pcolormesh([list(span), list(span)], cmap=cmap)
    plt.colorbar(psm, ax=ax)
    ax.set(title=title)
    assign_labels(ax)


def plot_wc_pacmap(
    X: anndata.AnnData,
    cmp: int,
    ax: Axes,
    cbarMax: float = 1.0,
    factor_matrix: str | None = None,
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
    values /= np.max(values)

    _plot_shaded_pacmap(
        points, values, ax, cmap, (-cbarMax, cbarMax), "Cmp. " + str(cmp)
    )


def plot_labels_pacmap(
    X: anndata.AnnData,
    labelType: str,
    ax: Axes,
    condition=None,
    cmap="tab20",
    color_key=None,
):
    """Scatterplot of PaCMAP visualization weighted by condition or cell type"""
    labels = X.obs[labelType]

    if condition is not None:
        labels = pd.Series([c if c in condition else "Z Other" for c in labels])
    if labels.dtype == "category":
        labels = labels.cat.set_categories(
            np.sort(labels.cat.categories.values), ordered=True
        )
    indices = np.argsort(labels)

    points = X.obsm["PaCMAP"][indices, :]
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

    ds_show(result, ax)
    ax.legend(handles=legend_elements)
    ax = assign_labels(ax)


def plot_wc_per_celltype(
    X: anndata.AnnData,
    cmp: int,
    ax: Axes,
    outliers: bool = False,
    cellType="cell_type",
    factor_matrix: str | None = None,
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
        ax=ax,
    )
    maxvalue = np.max(np.abs(ax.get_xticks()))
    ax.set(xticks=np.linspace(-maxvalue, maxvalue, num=5), xlabel="WC Weight")
    ax.set_title(cmpName)


def plot_lr_pacmap(lr: str, X: anndata.AnnData, ax: Axes, clip_outliers=0.9995):
    """Scatterplot of PaCMAP visualization weighted by lr"""
    lrList = X[:, lr].to_df().values

    lrList = np.clip(lrList, None, np.quantile(lrList, clip_outliers))
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

    values = lrList
    values -= np.min(values)
    values /= np.max(values)

    points = np.array(X.obsm["PaCMAP"])

    _plot_shaded_pacmap(points, values, ax, cmap, (0, 1), f"{lr}")


def assign_labels(ax):
    ax.set(xlabel="PaCMAP1", ylabel="PaCMAP2", xticks=[], yticks=[])
    return ax
