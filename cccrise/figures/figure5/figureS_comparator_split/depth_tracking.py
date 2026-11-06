"""Depth tracking of every method's components, both cohorts."""

from ... import common
from .._depth import depth_tracking

AXES = ("split_depth",)


def plot(axs):
    depth_tracking(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
