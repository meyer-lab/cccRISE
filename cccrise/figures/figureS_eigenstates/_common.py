"""Shared drawing for the eigen-state / component fine-cell-type histograms,
from ``uv run cccrise-eigenstate-celltypes`` (`validation.eigenstate_celltypes`).

Each score (E = RISE eigen-state projection, S / R = cccRISE component sender
/ receiver loading; sample-scaled, oriented) is drawn over the cells of its
home coarse types, stacked by fine type, with the active cut (half the
score's 99th percentile, tau = 0.5). The legend gives each fine type's active
fraction; the title gives the top fine type's at tau = 0.25 / 0.5 / 0.75. A
fraction that holds as the cut moves is a sub-population a cell-type
annotation cannot represent; one that moves with the cut is a graded level
across the type.
"""

from functools import cache

import matplotlib as mpl
import numpy as np
import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from .. import common, style

DATASET = "alad"
HINT = f"    uv run cccrise-eigenstate-celltypes --dataset {DATASET}"
CALL_LABELS = {
    "fine-aligned": "fine-aligned",
    "cross-cutting": "cross-cutting",
    "partial in one fine type": "partial, one fine type",
    "mixed": "diffuse",
}


@cache
def load(name: str) -> pd.DataFrame:
    """summary, profile, hist or null."""
    return common.load_runner_output(
        f"eigenstate_celltype_{name}_{DATASET}.parquet", VALIDATION_DIR, hint=HINT
    )


@cache
def fine_colors() -> dict[str, str]:
    """One fixed colour per fine type: the colourblind palette for the
    lymphoid / myeloid / other types, a second palette for the macrophage
    family (which only appear together with each other)."""
    profile = load("profile")[["coarse", "fine"]].drop_duplicates()
    macro = profile["coarse"].isin(["Macrophages"]) | profile["fine"].str.contains(
        "Mac"
    )
    others = sorted(profile.loc[~macro, "fine"])
    macros = sorted(profile.loc[macro, "fine"])
    first = [*style.CATEGORICAL, "0.35", "0.6"]
    second = [mpl.colors.to_hex(c) for c in mpl.colormaps["tab20b"].colors]
    return dict(zip(others, first, strict=False)) | dict(
        zip(macros, second, strict=False)
    )


def plot_unit(ax, unit: str, legend: bool = True) -> None:
    hist = load("hist")
    hist = hist[hist["unit"] == unit]
    summary = load("summary").set_index("unit").loc[unit]
    profile = load("profile")
    profile = profile[profile["unit"] == unit].set_index("fine")
    colors = fine_colors()
    fines = (
        hist.groupby("fine")["count"].sum().sort_values(ascending=False).index.tolist()
    )
    bottom = None
    for fine in fines:
        rows = hist[hist["fine"] == fine].sort_values("bin_left")
        counts = rows["count"].to_numpy(float)
        frac = profile.loc[fine, "active_frac"]
        ax.bar(
            rows["bin_left"],
            counts,
            width=rows["bin_right"] - rows["bin_left"],
            bottom=bottom,
            align="edge",
            color=colors.get(fine, "0.5"),
            linewidth=0,
            label=f"{fine} ({frac:.0%})",
        )
        bottom = counts if bottom is None else bottom + counts
    ax.axvline(hist["threshold"].iloc[0], color=style.COLORS["ink"], ls="--", lw=0.8)
    ax.set_title(
        f"{unit} ({CALL_LABELS[summary['call']]})\n"
        f"{summary['top_fine']} active at\n"
        r"$\tau$ = .25 / .5 / .75: "
        f"{summary['top_fine_active_frac_tau0.25']:.0%} / "
        f"{summary['top_fine_active_frac']:.0%} / "
        f"{summary['top_fine_active_frac_tau0.75']:.0%}",
        fontsize=style.FONT_SIZE_DENSE - 1,
    )
    ax.set_yscale("symlog", linthresh=10)
    ax.tick_params(labelsize=style.FONT_SIZE_DENSE - 1)
    ax.grid(False)
    if legend:
        ax.legend(
            fontsize=style.FONT_SIZE_DENSE - 1.5,
            frameon=False,
            loc="upper right",
            handlelength=0.8,
            title="fine type (active)",
            title_fontsize=style.FONT_SIZE_DENSE - 1.5,
        )


def plot_grid(fig, units: list[str], n_cols: int) -> None:
    n_rows = int(np.ceil(len(units) / n_cols))
    axes = np.atleast_1d(fig.subplots(n_rows, n_cols)).ravel()
    for ax, unit in zip(axes, units, strict=False):
        plot_unit(ax, unit)
    for ax in axes[len(units) :]:
        ax.set_axis_off()
    fig.supxlabel(
        "score (sample-scaled; dashed: active cut)", fontsize=style.FONT_SIZE_DENSE
    )
    fig.supylabel("cells", fontsize=style.FONT_SIZE_DENSE)
