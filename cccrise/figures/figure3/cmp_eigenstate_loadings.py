"""The outcome-associated CP component's sender and receiver loadings on each
RISE eigenstate, as Figure 2 shows for its granularity component.

An alternative to `celltype_association` + `sender_receiver_scatter` for
identifying the component's cell types (with the cell type x eigenstate
supplement); not in the figure's layout.
"""

import numpy as np

from .. import common
from ._common import COMPONENT, factors_adata

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    adata = factors_adata()
    sender, receiver = (np.asarray(adata.uns[k])[:, COMPONENT - 1] for k in "BC")
    largest = np.abs(np.column_stack([sender, receiver])).max(axis=1)
    common.plot_component_eigenstate_loadings(
        adata, COMPONENT, ax, [int(np.argmax(largest)) + 1]
    )
    ax.set_ylabel(f"Cmp. {COMPONENT} loading")


def makeFigure():
    return common.standalone_fixed(__name__)
