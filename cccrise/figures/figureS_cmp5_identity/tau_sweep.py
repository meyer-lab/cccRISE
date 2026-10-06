"""Size of each label-free group (`figure2.group_scatter`)
across the cutoff fraction τ, the chosen `TAU` marked."""

import numpy as np

from .. import common

TAUS = np.round(np.arange(0.05, 0.81, 0.05), 2)
AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    sender, receiver = common.component_weights()
    counts = [common.split_cells(sender, receiver, tau) for tau in TAUS]
    for group, color in common.GROUP_COLORS.items():
        ax.plot(
            TAUS,
            [c[group].sum() for c in counts],
            marker="o",
            markersize=2,
            color=color,
            label=group.capitalize(),
        )
    common.reference_line(ax, common.GROUP_TAU, horizontal=False)
    ax.set_yscale("log")
    ax.set_ylim(
        bottom=min(c[g].sum() for c in counts for g in common.GROUP_COLORS) / 1.5
    )
    ax.legend(loc="upper right")
    ax.set(xlabel="Cutoff fraction τ of max weight", ylabel="Cells")


def makeFigure():
    return common.standalone_fixed(__name__)
