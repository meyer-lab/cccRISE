"""Supplementary figure: NicheNet on cccRISE and every comparator column of ALAD
(``uv run cccrise-comparator-nichenet``). Each column's ligand-weighted target
score above rankings with shuffled cells (A), its ligand-weighted score percentile
against a shuffled prior (B), and each program against its closest columns (C)."""

from ... import common, style
from . import cell_shuffle, closest_columns, prior_shuffle

LAYOUT = [[cell_shuffle, prior_shuffle, closest_columns]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
