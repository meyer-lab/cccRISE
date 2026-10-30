"""IBD RISE rank selection by bi-cross-validation, from ``uv run cccrise-riserank``."""

from ... import common
from ...figure2.rise_rank_selection import draw

DATASET = "ibd"
AXES = ("rect",)


def plot(axs):
    # IBD's BiCV test curve runs along the bottom of the panel
    draw(axs[0], DATASET, key_loc="center right")


def makeFigure():
    return common.standalone_fixed(__name__)
