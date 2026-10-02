"""Comparison of CCC-RISE against the re-implemented comparator methods
(`cccrise.comparators`: NICHES, Scriabin, STACCato).

Placeholder: no runner produces comparator results yet.
"""

from .. import common, style


def plot(fig):
    common.placeholder(
        fig, "Comparison to NICHES / Scriabin / STACCato\n(needs a comparator runner)"
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
