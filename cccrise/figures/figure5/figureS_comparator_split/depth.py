"""Absolute Spearman correlation between each component's sample loadings and
median genes detected per cell, for every method's fit, ALAD and IBD cohorts,
with Kruskal-Wallis tests across fits and one-sided Wilcoxon rank-sum tests per fit.

How far every method's components track detection depth, ALAD beside IBD
(`_depth.depth_tracking`): |Spearman rho| of each component's sample loadings
with median genes detected per cell, a strip over a box per fit, with each
cohort's Kruskal-Wallis p and the fits that track depth more than the rest
starred."""

from ... import common
from .._depth import depth_tracking

AXES = ("split_depth",)


def plot(axs):
    depth_tracking(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
