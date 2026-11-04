"""Supplementary figure: which components associate with which ALAD outcome,
and whether the associated comparator columns are programs. Every
component's η² for 6-month outcome, 6-month death, 1-year death and acute
rejection, labelled with its FDR (A-D); each associated comparator column's
top pairs in it and its three closest columns of other fits, with |cos| and
Δρ (E-G); and the sample profiles of C4, P5 fine and S2 coarse, which
carries P5's pairs (H)."""

from ... import common, style
from . import (
    capture_p1,
    capture_p5,
    capture_s11,
    death_1yr,
    death_6mo,
    outcome,
    rejection,
    sample_profiles,
)

LAYOUT = [
    [outcome, death_6mo, death_1yr, rejection],
    [capture_p1, capture_p5, capture_s11],
    [sample_profiles],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
