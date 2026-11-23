"""Each EE, EC, and tuft cell's receiver weight on CPD component 3, on the
EE/EC/tuft PaCMAP."""

from ... import common
from . import _common as H

AXES = ("square",)


def plot(axs):
    H.receiver_map(axs, 3)


def makeFigure():
    return common.standalone_fixed(__name__)
