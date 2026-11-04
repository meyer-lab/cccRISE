"""How few sender eigenstates the hormone programs use against how many the
cells they come from span: the cumulative share of C3's and C4's squared
sender loadings, and of EE cells', EC cells' and the secretory lineage's
mean squared RISE projections (`run_ibd_programs.eigenstate_energy`), over
eigenstates from largest; the dotted line marks `SHARE`."""

import numpy as np
import pandas as pd

from ... import common, style
from .. import _common as S

AXES = ("square",)
SHOWN = 20
SHARE = 0.8
CELL_COLORS = {
    "EE cells": style.CATEGORICAL[2],
    "EC cells": style.CATEGORICAL[4],
    "Secretory lineage": "0.5",
}


def cumulative(energy: np.ndarray) -> np.ndarray:
    return np.cumsum(np.sort(energy)[::-1]) / energy.sum()


def plot(axs):
    (ax,) = axs
    B = S.factors()["cp_B"].values.astype(float)
    energy = S.programs("eigenstate_energy")
    curves = {
        f"C{k} sender": (cumulative(B[:, k - 1] ** 2), color, "-")
        for k, color in S.HORMONE_COLORS.items()
    }
    for name, color in CELL_COLORS.items():
        values = energy.loc[energy["Cells"] == name].sort_values("Eigenstate")[
            "Mean squared projection"
        ]
        curves[name] = (cumulative(values.to_numpy()), color, "--")
    common.source_data(
        "eigenstate_cumulative_share",
        pd.DataFrame({k: v[0] for k, v in curves.items()})
        .rename_axis("Eigenstates")
        .set_axis(pd.RangeIndex(1, B.shape[0] + 1, name="Eigenstates")),
        "Cumulative share of squared sender loadings (programs) or mean squared RISE "
        "projections (cells) over eigenstates, largest first.",
    )
    x = np.arange(1, SHOWN + 1)
    for label, (curve, color, ls) in curves.items():
        ax.plot(
            x,
            curve[:SHOWN],
            color=color,
            ls=ls,
            label=label,
            marker="o",
            markersize=style.MARKER_SIZE * 0.6,
        )
    common.reference_line(ax, SHARE)
    ax.set_xlim(0.5, SHOWN + 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Sender eigenstates")
    ax.set_ylabel("Cumulative share")
    ax.legend(loc="lower right", fontsize=style.FONT_SIZE_DENSE, handlelength=1.5)


def makeFigure():
    return common.standalone_fixed(__name__)
