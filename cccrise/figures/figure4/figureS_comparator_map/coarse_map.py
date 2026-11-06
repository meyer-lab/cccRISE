"""Figure 4A on coarse labels."""

from ... import common
from .._recovery import program_map

AXES = ("map_coarse",)
#: Labels of programs that sit together at the origin.
SPOTS = {
    ("ibd", 3): (0.0, 0.135, "left"),
    ("ibd", 4): (0.055, 0.135, "left"),
    ("ibd", 1): (0.105, 0.05, "left"),
}


def plot(axs):
    program_map(axs[0], "coarse", SPOTS, "Coarse labels")


def makeFigure():
    return common.standalone_fixed(__name__)
