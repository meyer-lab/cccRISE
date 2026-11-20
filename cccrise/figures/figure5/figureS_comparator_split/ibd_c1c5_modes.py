"""Cosine between IBD CCC-RISE components 1 and 5 in each CPD mode: sample,
sender eigen-state, receiver eigen-state, and LR pair loadings.

Why the comparators merge IBD C1 and C5: the cosine between the two
components in each CPD mode. They share their samples, sender eigenstates and
receiver eigenstates and differ only in their LR pairs, so a column fit on
pooled cell types sees one program."""

import numpy as np
import pandas as pd

from ....factors import load_factors_netcdf
from ....runners.datasets import DATASETS
from ... import common, style
from .._common import METHOD_COLORS

AXES = ("split_modes",)
MODES = {
    "cp_A": "Conditions",
    "cp_B": "Senders",
    "cp_C": "Receivers",
    "cp_D": "LR pairs",
}
PAIR = (1, 5)


def plot(axs):
    ax = axs[0]
    factors = load_factors_netcdf(DATASETS["ibd"].factors_path())
    i, j = (k - 1 for k in PAIR)
    cosines = {}
    for key, name in MODES.items():
        m = factors[key].to_numpy()
        cosines[name] = float(
            m[:, i] @ m[:, j] / np.linalg.norm(m[:, i]) / np.linalg.norm(m[:, j])
        )
    common.source_data(
        "ibd_c1_c5_mode_cosines",
        pd.Series(cosines, name="Cosine").rename_axis("Mode"),
        "Cosine between IBD components 1 and 5 in each CPD mode (sample, sender "
        "eigen-state, receiver eigen-state and LR pair loadings).",
    )
    y = np.arange(len(cosines))
    ax.barh(y, list(cosines.values()), color=METHOD_COLORS["cccRISE"], height=0.6)
    ax.axvline(0, color=style.COLORS["ink"], lw=0.6)
    ax.set(xlim=(-1, 1), xticks=[-1, 0, 1], ylim=(len(y) - 0.5, -0.5))
    ax.set_yticks(y, list(cosines), fontsize=style.FONT_SIZE_DENSE)
    ax.set_xlabel("Cosine, C1 vs. C5", fontsize=style.FONT_SIZE_DENSE)
    ax.tick_params(axis="x", labelsize=style.FONT_SIZE_DENSE)


def makeFigure():
    return common.standalone_fixed(__name__)
