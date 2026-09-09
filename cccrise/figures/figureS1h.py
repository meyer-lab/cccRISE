"""
Figure S1h: Decomposition of the pseudobulk communication data and tensor for BAL data.
"""

from typing import cast

import numpy as np
import pandas as pd

from ..tensor import run_pseudobulk_workflow
from .common import (
    get_covid_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors,
)
from .commonFuncs.plotGeneral import rotate_yaxis


def makeFigure():
    ax, f = getSetup((20, 8), (1, 4))
    subplotLabel(ax)

    X = get_covid_data(project_cells=True)
    X, _ = run_pseudobulk_workflow(X, cp_rank=8)
    groupby = "celltype"
    condition_column = "sample"
    group_col = "condition"
    sample_to_group = (
        cast(pd.DataFrame, X.obs)
        .drop_duplicates(subset=[condition_column, group_col])
        .set_index(condition_column)[group_col]
    )

    plot_condition_factors(
        data=X,
        ax=ax[0],
        cond="sample",
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
