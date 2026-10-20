"""Coordination: the effective number of LR pairs (`effective_pairs`) in each
LR column of each method; cccRISE's programs labeled."""

import numpy as np

from .. import common, style
from . import _common

AXES = ("rect",)
JITTER = 0.12


def columns():
    """(label, effective pairs per LR column), cccRISE first."""
    rows = [("cccRISE", _common.effective_pairs(_common.cccrise_lr()))]
    for method, annotation in _common.FITS:
        lr = _common.own_fits(annotation)[method].modes["lr"]
        rows.append((_common.fit_label(method, annotation), _common.effective_pairs(lr)))
    return rows


def _label_programs(ax, xs, values, gap: float = 0.25):
    """Label cccRISE's points C1..Ck, nudging labels apart where values sit
    within `gap` of each other."""
    placed: list[float] = []
    for k in np.argsort(values):
        y = values[k]
        while placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(f"C{k + 1}", (xs[k], values[k]), xytext=(xs[k] + 0.25, y),
                    textcoords="data", va="center", fontsize=style.FONT_SIZE_DENSE)  # fmt: skip


def plot(axs):
    (ax,) = axs
    rng = np.random.default_rng(0)
    for x, (label, values) in enumerate(columns()):
        color = style.COLORS["emphasis"] if label == "cccRISE" else style.COLORS["muted"]
        jitter = rng.uniform(-JITTER, JITTER, len(values))
        ax.scatter(x + jitter, values, s=style.POINT_SIZE, color=color, lw=0)
        if label == "cccRISE":
            _label_programs(ax, x + jitter, values)
    labels = [label for label, _ in columns()]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Effective LR pairs\nper component")


def makeFigure():
    return common.standalone_fixed(__name__)
