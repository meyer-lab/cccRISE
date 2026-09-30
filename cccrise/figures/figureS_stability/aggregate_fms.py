"""Aggregate FMS per stage and resampling axis (the Methods' stability
metric), from ``uv run cccrise-stability``."""

from .. import common, style
from .per_component_fms import load_stability


def plot(fig):
    ax = fig.subplots()
    common.plot_aggregate_fms(load_stability(), ax)
    ax.set_title("Aggregate stability")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
