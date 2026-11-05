"""Figure 4: what pooling cells by annotation costs. cccRISE against
pseudobulk CPD, Tensor-cell2cell and STACCato on ALAD, each comparator fit
on coarse and on fine labels: which cccRISE programs each comparator
contains (A); whether each component's top LR pairs move together across
samples, over all pairs (B) and gene-disjoint pairs (C); how much each
component tracks sequencing depth (D); which comparator columns cccRISE's
span contains (E) and what each LR column is (F); every component's
strongest outcome association (G); how the closest comparator columns
capture cccRISE's outcome program C4 (H); and, on ALAD and IBD, how much of
each program's per-cell weight any annotation could hold, and how much its
closest comparator column does (I, J)."""

from .. import common, style
from . import (
    classes,
    columns_in_cccrise,
    covariation,
    covariation_disjoint,
    depth,
    label_ceiling_receiver,
    label_ceiling_sender,
    program_capture,
    reach,
    strongest_association,
)

LAYOUT = [
    [reach, covariation, covariation_disjoint],
    [depth, columns_in_cccrise, classes],
    [strongest_association, program_capture],
    [label_ceiling_sender, label_ceiling_receiver],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
