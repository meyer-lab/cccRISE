"""
Figure A6c: cccRISE on BAL ALAD data with
"""

from typing import cast

import pandas as pd

from ...utils import correct_conditions
from ..common import (
    get_alad_data,
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

    X = get_alad_data(project_cells=True)
    condition_column = "dsco_id"
    X.uns["A"] = correct_conditions(X)

    group_col = "ALADstatus"
    sample_to_group = (
        cast(pd.DataFrame, X.obs)
        .drop_duplicates(subset=[condition_column, group_col])
        .set_index(condition_column)[group_col]
    )

    # Factor 0: Patient Conditions (Samples)
    plot_condition_factors(
        X,
        ax[0],
        cond=condition_column,
        cond_group_labels=sample_to_group,
        group_cond=True,  # Sort samples by their condition group
        normalize=True,
    )
    ax[0].set_title("Factor 0: Patient Condition")

    # Factor 1: Sender Cell Eigenstates
    plot_eigenstate_factors(X, ax[1], factor_type="B")
    ax[1].set_title("Factor 1: Sender Cell Eigen-state")
    rotate_yaxis(ax[1], rotation=0)

    # Factor 2: Receiver Cell Eigenstates
    plot_eigenstate_factors(X, ax[2], factor_type="C")
    ax[2].set_title("Factor 2: Receiver Cell Eigen-state")
    rotate_yaxis(ax[2], rotation=0)

    # Factor 3: Ligand-Receptor Pairs
    plot_lr_factors(X, ax[3], trim=True, weight=0.04)
    ax[3].set_title("Factor 3: LR Pair")

    return f
