"""R2X of the full-data RISE fit and of the bi-cross-validation training and
held-out blocks at each RISE rank, IBD cohort."""

from ... import common
from ...figure2.rise_rank_selection import draw

DATASET = "ibd"
AXES = ("square",)


def plot(axs):
    draw(axs[0], DATASET)


def makeFigure():
    return common.standalone_fixed(__name__)
