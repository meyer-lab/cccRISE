"""Supplementary figure: CCC-RISE against comparator methods."""

from .. import common, style
from . import method_comparison

PANELS = {"comparison": method_comparison}
LAYOUT = [["comparison"]]


def makeFigure():
    return common.compose(style.FIGSIZE["double_column"], LAYOUT, PANELS)
