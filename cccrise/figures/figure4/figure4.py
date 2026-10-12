"""Figure 4: Figure 2's model selection and validation panels for the IBDverse
Crohn's disease cohort -- RISE and CP rank selection, the CP factor matrices,
and NicheNet target validation."""

from .. import common, style
from ..figure2 import (
    cp_factor_matrices,
    cp_rank_selection,
    nichenet,
    rise_rank_selection,
)
from ._common import DATASET

LAYOUT = [
    [common.for_dataset(panel, DATASET) for panel in row]
    for row in (
        [rise_rank_selection, cp_rank_selection],
        [cp_factor_matrices],
        [nichenet],
    )
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
