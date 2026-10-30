"""Resolution: C5's per-cell sender against receiver weight (Fig. 2's
label-free sender and receiver groups colored), and the same cells' weights
pooled by sample x fine cell type -- the most of them any label-pooled tensor
can represent, since its entries are sample x type blocks."""

import numpy as np
import pandas as pd

from .. import common, style
from ..figure2 import _common as granularity
from . import _common

AXES = ("square", "square")


def pooled(values: np.ndarray) -> np.ndarray:
    """Each cell's value replaced by its sample x fine-type block's mean."""
    spec = common.DATASETS[_common.DATASET]
    obs = common.get_cell_projections(_common.DATASET).obs
    blocks = (obs[spec.condition_column].astype(str) + "|"
              + obs[_common.ANNOTATIONS["fine"][1]].astype(str)).to_numpy()  # fmt: skip
    return pd.Series(values).groupby(blocks).transform("mean").to_numpy()


def plot(axs):
    sender, receiver = granularity.component_weights()
    groups = granularity.cell_groups().to_numpy()
    for ax, (x, y), title in (
        (axs[0], (sender, receiver), "per cell (cccRISE)"),
        (axs[1], (pooled(sender), pooled(receiver)), "pooled by sample × type"),
    ):
        for group in ("rest", *granularity.GROUP_COLORS):
            mask = groups == group
            ax.scatter(x[mask], y[mask], s=style.POINT_SIZE_CELL if group == "rest" else 3,
                       c=granularity.GROUP_COLORS.get(group, style.COLORS["muted"]),
                       lw=0, rasterized=True,
                       label=f"{group.capitalize()} (n={mask.sum()})")  # fmt: skip
        ax.set_xlabel(f"C{granularity.GRANULARITY_COMPONENT} sender weight,\n{title}")
        ax.set_ylabel(f"C{granularity.GRANULARITY_COMPONENT} receiver weight")
    # one scale for both, so pooling's collapse reads directly
    low = min(min(ax.get_xlim()[0], ax.get_ylim()[0]) for ax in axs)
    high = max(max(ax.get_xlim()[1], ax.get_ylim()[1]) for ax in axs)
    for ax in axs:
        ax.set(xlim=(low, high), ylim=(low, high))
    common.legend(axs[1], "right", markerscale=2)


def makeFigure():
    return common.standalone_fixed(__name__)
