"""Figure 4: what pooling cells by annotation costs. cccRISE against
pseudobulk CPD, Tensor-cell2cell and STACCato on ALAD, each comparator fit
on coarse and on fine labels: which cccRISE programs each comparator
contains (A); whether each component's top LR pairs move together across
samples, over all pairs (B) and gene-disjoint pairs (C); how much each
component tracks sequencing depth (D); which comparator columns cccRISE's
span contains (E) and what each LR column is (F); every component's
strongest outcome association (G); and how the closest comparator columns
capture cccRISE's outcome program C4 (H)."""

from .. import common, style
from . import (
    classes,
    columns_in_cccrise,
    covariation,
    covariation_disjoint,
    depth,
    program_capture,
    reach,
    strongest_association,
)

LAYOUT = [
    [reach, covariation, covariation_disjoint],
    [depth, columns_in_cccrise, classes],
    [strongest_association, program_capture],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
