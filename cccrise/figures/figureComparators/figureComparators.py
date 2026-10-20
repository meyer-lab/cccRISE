"""The comparator figure: what pooling by annotation costs. Which of cccRISE's
ALAD programs label-pooled tensor methods recover, how much of a program's
per-cell structure a label can carry, how many LR pairs a program
coordinates, and which methods find an outcome association."""

from .. import common, style
from . import clinical, coordination, recovery, representability, resolution, schematic

LAYOUT = [
    [schematic, recovery],
    [representability, resolution],
    [coordination, clinical],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
