"""Association of every RISE component's condition loading (`pf2_A`) with
every ALAD outcome tested against the CP components (one-way ANOVA
eta-squared, FDR-corrected across components within an association)."""

from .. import common
from ..figure3 import _common as S

AXES = ("wide",)


def plot(axs):
    (ax,) = axs
    common.plot_association_heatmap(
        S.rise_association_table(), ax, xlabel="RISE component"
    )


def makeFigure():
    return common.standalone_fixed(__name__)
