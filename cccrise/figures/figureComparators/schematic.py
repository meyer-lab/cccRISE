"""Schematic: pooled methods take sender and receiver from the annotation;
cccRISE takes them from cells, through RISE eigen-states."""

from .. import common

AXES = ("rect",)


def plot(axs):
    common.placeholder(
        axs[0], "Labels → pooled tensor\nvs\ncells → eigen-states → tensor"
    )


def makeFigure():
    return common.standalone_fixed(__name__)
