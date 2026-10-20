"""How well inflamed and healthy samples separate: cross-validated AUROC from
composition alone and from cccRISE's sample factors."""

from .. import common, style
from . import _common

AXES = ("short",)


def plot(axs):
    (ax,) = axs
    values = _common.aurocs()
    labels = list(values)
    means = [values[k]["mean"] for k in labels]
    errors = [values[k]["se"] for k in labels]
    colors = [style.COLORS["emphasis"] if k.startswith("cccRISE") else style.COLORS["muted"] for k in labels]
    ax.barh(range(len(labels)), means, xerr=errors, color=colors, lw=0)
    ax.set_yticks(range(len(labels)), [k if k.startswith("cccRISE") else k[0].upper() + k[1:] for k in labels])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1)
    ax.set_xlabel("CV AUROC, inflamed vs healthy")


def makeFigure():
    return common.standalone_fixed(__name__)
