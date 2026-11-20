"""Association of every IBD CP component with every inflammation-status
association (one-way ANOVA eta-squared, FDR-corrected across all
associations and components), as Figure 3 tests ALAD's."""

from ... import common
from .. import _common as S
from ._common import save_association_data

AXES = ("mid",)
FDR = "Benjamini-Hochberg FDR-corrected across all associations and components"


def plot(axs):
    (ax,) = axs
    table = S.association_table("cp_A")
    common.plot_association_heatmap(table, ax)
    save_association_data(
        table,
        S.corrected_loadings("cp_A"),
        "component",
        "each CPD component's condition loading, read-depth corrected as tested",
        FDR,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
