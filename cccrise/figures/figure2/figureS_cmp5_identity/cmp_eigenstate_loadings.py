"""The granularity CP component's sender and receiver loadings on each RISE
eigenstate."""

import numpy as np
import pandas as pd

from ....runners.datasets import DATASETS
from ... import common
from .. import _common as S

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    adata = common.load_factors_to_anndata(
        DATASETS[S.GRANULARITY_DATASET].factors_path()
    )
    common.plot_component_eigenstate_loadings(
        adata,
        S.GRANULARITY_COMPONENT,
        ax,
        (S.GRANULARITY_SENDER_EIGENSTATE, S.GRANULARITY_RECEIVER_EIGENSTATE),
    )
    ax.set_ylabel(f"Cmp. {S.GRANULARITY_COMPONENT} loading")
    sender, receiver = (
        np.asarray(adata.uns[k])[:, S.GRANULARITY_COMPONENT - 1] for k in "BC"
    )
    common.source_data(
        "cp_eigenstate_loadings",
        pd.DataFrame(
            {
                "RISE eigenstate": np.arange(1, len(sender) + 1),
                "Sender loading (B)": sender,
                "Receiver loading (C)": receiver,
            }
        ),
        f"CP component {S.GRANULARITY_COMPONENT}'s sender (B) and receiver (C) "
        "factor loading on each RISE eigenstate.",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
