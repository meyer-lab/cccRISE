"""Aggregate RISE and CPD stability per resampling axis (the Methods'
stability metric), from ``uv run cccrise-stability``."""

from .. import common, style

DATASET = "alad"


def plot(fig):
    ax = fig.subplots()
    common.plot_aggregate_fms(common.load_stability(DATASET), ax)
    ax.set_title("Component stability")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"])
