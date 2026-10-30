"""Exploration of which cells carry cccRISE's IBD programs, against the fine
labels (`HORMONE_CELLS_PREREGISTRATION.md`). Each module is a standalone
figure; this master draws the support levels as the summary."""

from . import _results


def makeFigure():
    return _results.levels_figure()
