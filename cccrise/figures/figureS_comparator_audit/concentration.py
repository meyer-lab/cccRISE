"""Each missed column's top-pair share of its LR weight against its effective
number of samples, colored by class."""

from ...runners.run_comparator_audit import ONE_SAMPLE
from .. import common, style
from . import _common

AXES = ("square",)


def plot(axs):
    (ax,) = axs
    table = _common.audit()
    for name, color in _common.CLASSES.items():
        rows = table[table["class"] == name]
        ax.scatter(rows["effective_samples"], rows["top_share"], s=style.POINT_SIZE, color=color, lw=0)
    ax.axvline(ONE_SAMPLE, color=style.COLORS["reference"], ls=style.LINE_STYLE_REFERENCE,
               lw=style.LINE_WIDTH_THIN)  # fmt: skip
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Effective samples")
    ax.set_ylabel("Top pair's share\nof LR weight")


def makeFigure():
    return common.standalone_fixed(__name__)
