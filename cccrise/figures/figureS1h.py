"""
Figure S1h: Decomposition of the pseudobulk communication data and tensor for IBD data.

TODO(IBD): VERIFY. The pseudobulk CP rank (`PSEUDOBULK_RANK`) was chosen for the
retired BALF cohort; re-select it for IBD.
"""

from typing import cast

import numpy as np
import pandas as pd

from ..tensor import run_pseudobulk_workflow
from .common import (
    IBD_SPEC,
    get_ibd_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors,
)
from .commonFuncs.plotGeneral import rotate_yaxis

PSEUDOBULK_RANK = 8


def makeFigure():
    ax, f = getSetup((20, 8), (1, 4))
    subplotLabel(ax)

    X = get_ibd_data(project_cells=True)
    groupby = IBD_SPEC.celltype_column
    condition_column = IBD_SPEC.condition_column
    group_col = IBD_SPEC.group_column
    X, _ = run_pseudobulk_workflow(
        X,
        cp_rank=PSEUDOBULK_RANK,
        condition_column=condition_column,
        groupby=groupby,
    )
    sample_to_group = (
        cast(pd.DataFrame, X.obs)
        .drop_duplicates(subset=[condition_column, group_col])
        .set_index(condition_column)[group_col]
    )

    plot_condition_factors(
        data=X,
        ax=ax[0],
        cond=condition_column,
        cond_group_labels=sample_to_group,
        group_cond=True,
    )
    plot_eigenstate_factors(data=X, ax=ax[1], factor_type="B")

    celltype_names = np.unique(X.obs[groupby])
    ax[1].set_yticklabels(celltype_names)
    rotate_yaxis(ax[1], 0)

    plot_eigenstate_factors(
        data=X,
        ax=ax[2],
        factor_type="C",
    )
    ax[2].set_yticklabels(celltype_names)
    rotate_yaxis(ax[2], 0)

    plot_lr_factors(data=X, ax=ax[3], weight=0.03)

    return f
