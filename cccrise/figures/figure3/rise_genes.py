"""Every gene's loading on the outcome-associated RISE component (pf2_C)
against its rank, the top `MARKED_FRACTION` of genes marked and the
`N_LABELED` most heavily loaded named."""

import numpy as np

from ...factors import load_factors_netcdf
from .. import common
from . import _common as S

MARKED_FRACTION = 0.05  # of genes, marked on the curve
N_LABELED = 30  # as many of the top genes as the label columns hold
AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    table = S.rise_association_table()
    rows = table[table["Association"] == S.OUTCOME]
    component = int(rows.loc[rows["PAdj"].idxmin(), "Component"])
    factor = load_factors_netcdf(S.SPEC.factors_path())["pf2_C"]
    common.plot_ranked_loadings(
        ax,
        factor.values[:, component - 1].astype(float),
        np.asarray(factor["gene"].values).astype(str),
        round(MARKED_FRACTION * factor.shape[0]),
        N_LABELED,
    )
    ax.set(xlabel="Gene rank", ylabel=f"RISE component {component}\ngene loading")


def makeFigure():
    return common.standalone_fixed(__name__)
