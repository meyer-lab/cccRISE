"""Component match ambiguity: matched against runner-up FMS, from
``uv run cccrise-stability``."""

from .. import common, style
from .per_component_fms import load_stability


def plot(fig):
    ax = fig.subplots()
    common.plot_match_ambiguity(load_stability(), ax)
    ax.set_title("Match ambiguity")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
