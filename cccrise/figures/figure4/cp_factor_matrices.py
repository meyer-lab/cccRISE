"""CPD factor matrices of the IBD fit: each component's biopsy, sender eigen-
state, receiver eigen-state, and LR pair loadings, each divided by the
component's largest absolute loading, biopsies grouped by inflammation status.

The IBD CPD factor matrices, as Figure 2 draws ALAD's
(`figure2.cp_factor_matrices`), biopsies grouped by inflammation status, each
status named at a bracket and each LR label placed clear of the others
(`compact`)."""

from .. import common
from ..figure2.cp_factor_matrices import AXES, draw  # noqa: F401  (AXES: this panel's)
from ._common import DATASET, GROUP_COLORS, GROUP_COLUMN, SPEC, samples

EIGENSTATE_TICKS = (1, 15, 30, 45, 60)
# IBD's components are each carried by a handful of LR pairs: the top 2% (27
# rows) shows every one of them legibly, where more rows are mostly blank
TOP_LR_FRACTION = 0.02
STRIPS = {GROUP_COLUMN: ("Inflammation status", GROUP_COLORS)}


def plot(axs):
    groups = samples().set_index(samples()[SPEC.condition_column].astype(str))
    draw(axs, DATASET, groups, STRIPS, EIGENSTATE_TICKS, TOP_LR_FRACTION, compact=True)


def makeFigure():
    return common.standalone_fixed(__name__)
