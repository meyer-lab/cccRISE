"""How far each component's sample loadings track detection depth, every method."""

import numpy as np
import pandas as pd

from .. import common, style
from ._classes import cohort_bars
from ._common import DATASET_LABELS, METHOD_COLORS, load
from ._programs import FITS

BAR_DROP = 10.5


def depth_tracking(ax) -> None:
    """|rho| of each component's |sample profile| with median genes detected per
    cell, cccRISE then each comparator fit, ALAD beside IBD."""
    rng = np.random.default_rng(0)
    x, positions, names, spans, frames = 0.0, [], [], {}, []
    for dataset, label in DATASET_LABELS.items():
        start = x
        for method, lab in [("cccRISE", "coarse"), *FITS]:
            t = load("depth", lab, dataset)
            values = t.loc[t["Method"] == method, "AbsRho"].dropna().to_numpy()
            frames.append(
                pd.DataFrame(
                    {
                        "Cohort": label,
                        "Method": method,
                        "Labelling": lab,
                        "|rho|": values,
                    }
                )
            )
            ax.scatter(
                x + rng.uniform(-0.17, 0.17, len(values)),
                values,
                s=7,
                color=METHOD_COLORS[method],
                lw=0,
                zorder=3,
                marker="^" if lab == "fine" and method != "cccRISE" else "o",
            )
            ax.plot(
                [x - 0.27, x + 0.27],
                [np.median(values)] * 2,
                color=style.COLORS["ink"],
                lw=0.8,
                zorder=4,
            )
            positions.append(x)
            names.append(
                "cccRISE" if method == "cccRISE" else f"{method} {lab[0].upper()}"
            )
            x += 1
        spans[dataset] = (start, x - 1)
        x += 0.7
    common.source_data(
        "depth_tracking_by_component",
        pd.concat(frames, ignore_index=True),
        "|Spearman correlation| between each component's |sample profile| and "
        "per-sample depth (median genes detected per cell); lines are each fit's median.",
    )
    ax.set_xticks(
        positions,
        names,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=style.FONT_SIZE_DENSE - 0.7,
    )
    ax.set(ylim=(0, 1), yticks=[0, 0.5, 1], ylabel="Depth tracking, |ρ|")
    cohort_bars(ax, spans, BAR_DROP)
