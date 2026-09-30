"""
Figure S4m: Violin plot of a component's values for its top receiver cells.

TODO(IBD): NEEDS CHANGING OR REMOVAL. This figure was written for the retired
BALF cohort, which the IBD cohort supersedes. Its data loader now points at
IBD, but the component and cell-type column below were picked for the old
cohort and do not carry over: re-pick them for IBD, or delete this figure.
"""

import numpy as np
import seaborn as sns

from ..utils import (
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
)
from .common import (
    get_ibd_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotGeneral import rotate_xaxis


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = get_ibd_data(project_cells=True)
    ccc_rise_cmp = 6

    X_mdc_receiver = add_obs_cmp_label(
        X, cmp=ccc_rise_cmp, pos=True, top_perc=1, type="receiver"
    )
    X_mdc_receiver = add_obs_cmp_unique_one(X_mdc_receiver, cmp=ccc_rise_cmp)
    X_mdc_receiver = X_mdc_receiver[X_mdc_receiver.obs["Label"] != "NoLabel"]

    # Alter order based on factor value low to high
    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]
    print("Receiver cells:", X_mdc_receiver.shape)

    # Datafame of the component values for the receiver cells and their cell types
    df = X_mdc_receiver.obs[["celltype"]]
    print(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])

    df["receiver"] = np.array(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    print(df)
    sns.violinplot(data=df, x="celltype", y="receiver", ax=ax[1])
    rotate_xaxis(ax[1], 90)

    return f
