"""Supplementary figure: the IBD RISE factor matrices (Figure 2's
`figureS_rise_factor_matrices`)."""

from ... import common
from ...figure2.figureS_rise_factor_matrices import factors, genes
from .._common import DATASET

LAYOUT = [[common.for_dataset(factors, DATASET)], [common.for_dataset(genes, DATASET)]]


def makeFigure():
    return common.compose_fixed(LAYOUT)
