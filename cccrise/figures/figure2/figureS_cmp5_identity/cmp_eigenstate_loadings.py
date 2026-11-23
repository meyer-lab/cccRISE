"""CPD component 6's sender and receiver loadings on each RISE eigen-state."""

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
    sender, receiver = (
        np.asarray(adata.uns[k])[:, S.GRANULARITY_COMPONENT - 1] for k in "BC"
    )
    eigenstates = np.arange(1, len(sender) + 1)
    common.plot_component_eigenstate_loadings(
        adata, S.GRANULARITY_COMPONENT, ax, eigenstates.tolist()
    )
    ax.set_ylabel(f"Cmp. {S.GRANULARITY_COMPONENT} loading")
    common.source_data(
        "cpd_eigenstate_loadings",
        pd.DataFrame(
            {
                "RISE eigen-state": eigenstates,
                "Sender loading (B)": sender,
                "Receiver loading (C)": receiver,
            }
        ),
        f"CPD component {S.GRANULARITY_COMPONENT}'s sender (B) and receiver (C) "
        "factor loading on each RISE eigen-state.",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
