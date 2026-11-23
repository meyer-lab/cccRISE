"""Each EE, EC, and tuft cell's receiver weight on CPD component 4, on the
EE/EC/tuft PaCMAP."""

from ... import common
from . import _common as H

AXES = ("square",)


def plot(axs):
    H.receiver_map(axs, 4)


def makeFigure():
    return common.standalone_fixed(__name__)
