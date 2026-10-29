"""Association of every CP component with every ALAD outcome (one-way ANOVA
eta-squared, FDR-corrected across all associations and components)."""

import pandas as pd

from ....utils import correct_conditions
from ... import common
from .. import _common as S
from ._common import save_association_data

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    table = S.association_table()
    common.plot_association_heatmap(table, ax)
    loadings = correct_conditions(S.factors_adata())
    save_association_data(
        table,
        pd.DataFrame(loadings, columns=range(1, loadings.shape[1] + 1)),
        "component",
        "each CP component's condition loading, read-depth corrected as tested",
        "is Benjamini-Hochberg FDR-corrected across all associations and components",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
