"""Supplementary figure (after Figure 4): on IBD, inflammation is largely a
shift in cell composition, and no method's association with it survives
adjustment for composition."""

from .. import common
from . import association, celltypes, separability

LAYOUT = [[celltypes, separability], [association]]


def makeFigure():
    return common.compose_fixed(LAYOUT)
