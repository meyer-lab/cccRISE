"""Figure 5B: each CCC-RISE component's containment in each comparator fit's
LR span, ||P_S c||^2 (`comparators.scores.containment`): the fraction of the
component's LR pair loadings the best combination of the fit's LR columns
reconstructs."""

import pandas as pd

from .. import common
from ._common import DATASET_LABELS, load
from ._programs import FITS
from ._recovery import colorbar_right, fit_grid

AXES = ("fig4_best",)
LABEL = "Fraction of LR pair\nloadings reconstructed"


def rows():
    frames = []
    for dataset, label in DATASET_LABELS.items():
        t = load("containment", dataset=dataset)
        t = t[(t["Of"] == "cccRISE") & (t["Mode"] == "lr")]
        frames.append(t[t["In"].isin({m for m, _ in FITS})].assign(Dataset=label))
    return pd.concat(frames, ignore_index=True)


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
    colorbar_right(ax, LABEL)


def makeFigure():
    return common.standalone_fixed(__name__)
