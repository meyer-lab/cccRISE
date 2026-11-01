"""IBD RISE rank selection by bi-cross-validation, from ``uv run cccrise-riserank``."""

from ... import common
from ...figure2.rise_rank_selection import draw

DATASET = "ibd"
AXES = ("square",)


def plot(axs):
    draw(axs[0], DATASET)


def makeFigure():
    return common.standalone_fixed(__name__)
