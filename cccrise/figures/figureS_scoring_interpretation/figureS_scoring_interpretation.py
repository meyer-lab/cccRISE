"""Supplementary figure: interpretation of CP components under each
communication score."""

from .. import common, style
from . import component_loadings

PANELS = {"loadings": component_loadings}
LAYOUT = [["loadings"]]


def makeFigure():
    return common.compose(style.FIGSIZE["full_page"], LAYOUT, PANELS)
