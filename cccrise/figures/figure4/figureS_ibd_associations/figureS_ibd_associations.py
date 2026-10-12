"""Supplementary figure: association of every CP component and every RISE
component with the IBD sample variables (`figures.associations`), as Figure 3's
`figureS_alad_associations` tests ALAD's."""

from ... import common
from ...figure3.figureS_alad_associations import cp_associations, rise_associations
from .._common import DATASET

LAYOUT = [
    [common.for_dataset(cp_associations, DATASET)],
    [common.for_dataset(rise_associations, DATASET)],
]


def makeFigure():
    return common.compose_fixed(LAYOUT)
