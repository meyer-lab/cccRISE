"""Containment of each ALAD CCC-RISE component's LR pair loadings in each
comparator fit's LR column span: the fraction of the loadings reconstructed by
the best linear combination of the fit's LR columns.

Figure 5B: each ALAD CCC-RISE component's containment in each comparator
fit's LR span, ||P_S c||^2 (`comparators.scores.containment`): the fraction
of the component's LR pair loadings the best combination of the fit's LR
columns reconstructs. IBD's is in `figureS_comparator_ibd_and_depth`."""

import pandas as pd

from .. import common
from ._common import DATASET_LABELS, load
from ._programs import FITS
from ._recovery import colorbar_beneath, fit_grid

AXES = ("fig5_best",)
LABEL = "Fraction of LR pair\nloadings reconstructed"


def rows(datasets=tuple(DATASET_LABELS)) -> pd.DataFrame:
    """CCC-RISE's LR containment in each comparator fit, per cohort."""
    frames = []
    for dataset in datasets:
        t = load("containment", dataset=dataset)
        frames.append(
            t[t["In"].isin({m for m, _ in FITS})].assign(
                Dataset=DATASET_LABELS[dataset]
            )
        )
    return pd.concat(frames, ignore_index=True)


def containment_grid(ax, datasets=tuple(DATASET_LABELS)) -> None:
    """Each component's containment in each fit, the cohorts of `datasets`
    one under another; no colorbar."""
    t = rows(datasets)
    common.source_data(
        "component_containment_in_comparator_span",
        t.rename(columns={"In": "Fit method"})[
            ["Dataset", "Component", "Fit method", "Labelling", "Containment"]
        ],
        "Each CCC-RISE component's LR pair loadings' containment in each comparator "
        "fit's LR column span: the squared norm of the unit loading's projection, the "
        "fraction of the loadings the fit's best combination of columns reconstructs.",
    )
    fit_grid(
        ax,
        {
            dataset: t[t["Dataset"] == DATASET_LABELS[dataset]].pivot_table(
                index="Component", columns=["In", "Labelling"], values="Containment"
            )
            for dataset in datasets
        },
    )


def plot(axs):
    containment_grid(axs[0], ("alad",))
    colorbar_beneath(axs[0], LABEL)


def makeFigure():
    return common.standalone_fixed(__name__)
