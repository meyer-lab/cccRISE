"""Largest absolute cosine between the LR pair loadings of two pseudobulk CPD
components, per rank and seeded SVD start, ALAD and IBD cohorts, fine cell type
labels."""

from ... import common
from ._panels import pb_collinearity

AXES = ("square",)


def plot(axs):
    pb_collinearity(axs, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
