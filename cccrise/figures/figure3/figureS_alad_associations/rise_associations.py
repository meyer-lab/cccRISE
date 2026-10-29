"""Association of every RISE component's condition loading (`pf2_A`) with
every ALAD outcome tested against the CP components (one-way ANOVA
eta-squared, FDR-corrected across all associations and components)."""

import copy

import pandas as pd

from ....factors import load_factors_netcdf
from ....utils import correct_conditions
from ... import common
from .. import _common as S
from ._common import save_association_data

AXES = ("wide",)


def plot(axs):
    (ax,) = axs
    table = S.rise_association_table()
    common.plot_association_heatmap(table, ax, xlabel="RISE component")
    # The read-depth-corrected pf2_A that `S.rise_association_table` tests
    adata = copy.copy(S.factors_adata())
    pf2_A = load_factors_netcdf(S.SPEC.factors_path())["pf2_A"].values
    adata.uns = {**adata.uns, "A": pf2_A}
    loadings = correct_conditions(adata)
    save_association_data(
        table,
        pd.DataFrame(loadings, columns=range(1, loadings.shape[1] + 1)),
        "component",
        "each RISE component's condition loading (pf2_A), read-depth corrected "
        "as tested",
        "is Benjamini-Hochberg FDR-corrected across all associations and components",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
