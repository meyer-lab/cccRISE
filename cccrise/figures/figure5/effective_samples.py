"""How many samples carry each CP component, in IBD and in ALAD: the
participation ratio of the component's raw condition loadings, (sum a^2)^2 /
sum a^4 -- 1 when one sample carries it all, the cohort's size when every
sample carries it equally."""

import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from ...factors import load_factors_netcdf
from ...runners.datasets import DATASETS
from .. import common, style

#: (registry key, label), in plotting order
COHORTS = (("alad", "ALAD"), ("ibd", "IBD"))
COHORT_GAP = 1  # empty slots between the cohorts' bars
AXES = ("rect",)


def effective_samples(dataset: str) -> pd.DataFrame:
    """Each CP component's participation ratio over `dataset`'s samples."""
    A = load_factors_netcdf(DATASETS[dataset].factors_path())["cp_A"].values
    A = A.astype(float) ** 2
    ratio = A.sum(axis=0) ** 2 / (A**2).sum(axis=0)
    return pd.DataFrame(
        {
            "Component": np.arange(1, len(ratio) + 1),
            "Effective samples": ratio,
            "Samples": len(A),
        }
    )


def plot(axs):
    (ax,) = axs
    tables = {label: effective_samples(key) for key, label in COHORTS}
    common.source_data(
        "effective_samples",
        pd.concat(
            [t.assign(Cohort=label) for label, t in tables.items()], ignore_index=True
        ),
        "Effective number of samples carrying each CP component: the "
        "participation ratio (sum a^2)^2 / sum a^4 of its raw condition "
        "loadings a, out of the cohort's samples.",
    )
    below = blended_transform_factory(ax.transData, ax.transAxes)
    ticks, labels, start = [], [], 0
    for color, (label, table) in zip(style.CATEGORICAL, tables.items(), strict=False):
        x = start + np.arange(len(table))
        values = table["Effective samples"]
        ax.bar(x, values, color=color, width=0.7)
        for xi, value in zip(x, values, strict=True):
            ax.text(
                xi,
                value,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=style.FONT_SIZE_DENSE,
            )
        n = int(table["Samples"].iloc[0])
        ax.hlines(
            n,
            x[0] - 0.4,
            x[-1] + 0.4,
            colors=style.COLORS["reference"],
            linestyles="--",
            lw=style.LINE_WIDTH_THIN,
        )
        ax.text(
            x.mean(), -0.16, f"{label} (n={n})", transform=below, ha="center", va="top"
        )
        ticks += list(x)
        labels += [str(c) for c in table["Component"]]
        start = x[-1] + 1 + COHORT_GAP
    ax.set_xticks(ticks, labels)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter("{x:g}")
    ax.set_xlim(-0.6, ticks[-1] + 0.6)
    ax.set_xlabel("CP component", labelpad=11)
    ax.set_ylabel("Effective samples")


def makeFigure():
    return common.standalone_fixed(__name__)
