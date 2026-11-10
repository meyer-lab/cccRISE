"""Figure 5B: each CCC-RISE component's containment in each comparator fit's
LR span, ||P_S c||^2 (`comparators.scores.containment`): the share of the
component's LR loading the best combination of the fit's LR columns
rebuilds."""

import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .. import common, style
from ._common import DATASET_LABELS, load
from ._programs import FITS
from ._recovery import HEAT, fit_grid

AXES = ("fig4_best",)
LABEL = "Containment in\ncomparator LR span"


def rows():
    frames = []
    for dataset, label in DATASET_LABELS.items():
        t = load("containment", dataset=dataset)
        t = t[(t["Of"] == "cccRISE") & (t["Mode"] == "lr")]
        frames.append(t[t["In"].isin({m for m, _ in FITS})].assign(Dataset=label))
    return pd.concat(frames, ignore_index=True)


def _colorbar_right(ax) -> None:
    """A vertical colorbar right of the cohort labels."""
    width = ax.bbox.width / ax.figure.dpi
    cax = ax.inset_axes((1 + 7 * style.MM / width, 0.2, 2 * style.MM / width, 0.6))
    ax.figure.colorbar(ScalarMappable(Normalize(0, 1), HEAT), cax=cax)
    cax.set_yticks([0, 1])
    cax.tick_params(labelsize=style.FONT_SIZE_DENSE, length=1.5, pad=1)
    cax.set_ylabel(LABEL, fontsize=style.FONT_SIZE_DENSE, labelpad=1)


def plot(axs):
    ax = axs[0]
    t = rows()
    common.source_data(
        "component_containment_in_comparator_span",
        t.rename(columns={"In": "Fit method", "NullQ95": "Row-permutation null q95"})[
            [
                "Dataset",
                "Component",
                "Fit method",
                "Labelling",
                "Containment",
                "Row-permutation null q95",
            ]
        ],
        "Each CCC-RISE component's LR loading's containment in each comparator fit's "
        "LR column span (squared norm of its projection, unit loading), with the 95th "
        "percentile of that containment with the span's LR pairs permuted.",
    )
    fit_grid(
        ax,
        {
            dataset: t[t["Dataset"] == label].pivot_table(
                index="Component", columns=["In", "Labelling"], values="Containment"
            )
            for dataset, label in DATASET_LABELS.items()
        },
    )
    _colorbar_right(ax)


def makeFigure():
    return common.standalone_fixed(__name__)
