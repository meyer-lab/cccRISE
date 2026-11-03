"""C4's, P5 fine's and S2 coarse's sample profiles by 6-month outcome."""

from ... import common
from .._columns import sample_profiles

AXES = ("full",)


def plot(axs):
    sample_profiles(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
