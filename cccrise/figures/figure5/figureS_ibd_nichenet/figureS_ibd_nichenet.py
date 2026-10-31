"""Supplementary figure: NicheNet downstream-target validation of the IBD CP
components, and the per-ligand scores and percentiles behind it, as Figure 2
and its NicheNet supplement show ALAD's."""

from ... import common, style
from . import ligand_scores, nichenet, weighted_percentiles

LAYOUT = [[nichenet], [ligand_scores, weighted_percentiles]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
