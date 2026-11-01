"""Each annotation-pooled comparator's own rank selection on one labelling,
from ``uv run cccrise-comparators``: pseudobulk CPD by cccRISE's seeded-start
sweep (`cp_rank`'s panels), Tensor-cell2cell by the Kneedle elbow of its
error (with how nearly two of a start's components duplicate each other, and
`cp_rank`'s minima of its random starts), and STACCato by each mode's
eigenvalue shares above 1%."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ....comparators import methods
from ....runners.datasets import (
    COMPARATOR_FITS_DIR,
    COMPARATOR_RANKS_DIR,
    comparator_filename,
)
from ....runners.provenance import read_metadata
from ... import common, cp_rank, style

DATASET = "alad"
HINT = "    uv run cccrise-comparators --dataset <dataset>"
STAC_MODES = {
    "sample": "Sample",
    "lr": "LR pair",
    "sender": "Sender",
    "receiver": "Receiver",
}
#: Leading eigenvalue shares drawn per STACCato mode.
STAC_COMPONENTS = 30


def load(method: str, what: str, dataset: str, labelling: str) -> pd.DataFrame:
    return common.load_runner_output(
        comparator_filename(method, what, dataset, labelling),
        COMPARATOR_RANKS_DIR,
        HINT,
    )


def fit_rank(method: str, dataset: str, labelling: str) -> int:
    """The rank `method`'s fit on `labelling` was written at."""
    path = COMPARATOR_FITS_DIR / comparator_filename(method, "fit", dataset, labelling)
    with xr.open_dataset(path, engine="h5netcdf") as fit:
        return int(fit.attrs["rank"])


#: Each Tensor-cell2cell variant's title.
TC2C_NAMES = {"tc2c": "TC2C", "tc2c_inner": "TC2C (published)"}


def tc2c_starts(dataset: str, labelling: str, method: str = "tc2c") -> pd.DataFrame:
    """A Tensor-cell2cell variant's runs as a `cp_rank` sweep (its runs as
    'Start')."""
    elbow = load(method, "elbow", dataset, labelling)
    return elbow.rename(columns={"Run": "Start"})


def title(
    ax: plt.Axes,
    name: str,
    dataset: str,
    labelling: str,
    sweep: tuple[str, str] = ("pb", "rank"),
) -> None:
    """`name` and the number of cell types in the tensor of `sweep` (method,
    table), from its provenance."""
    method, what = sweep
    path = COMPARATOR_RANKS_DIR / comparator_filename(method, what, dataset, labelling)
    n_types = len(read_metadata(path)["tensor"]["celltypes"])
    ax.set_title(f"{name}, {labelling} ({n_types} types)", loc="left")


def band(ax: plt.Axes, by_rank, center: pd.Series, **kwargs) -> None:
    """`center` by rank, banded by `by_rank`'s range."""
    color = style.COLORS["fit"]
    ax.fill_between(
        center.index, by_rank.min(), by_rank.max(), color=color, alpha=0.2, lw=0
    )
    ax.plot(center.index, center, color=color, **kwargs)


def pb_fit(axs, dataset: str, labelling: str) -> None:
    cp_rank.plot_r2x_starts(
        axs[0],
        load("pb", "rank", dataset, labelling),
        fit_rank("pb", dataset, labelling),
    )
    title(axs[0], "PB", dataset, labelling)


def pb_collinearity(axs, dataset: str, labelling: str) -> None:
    cp_rank.plot_mode_collinearity(
        axs[0],
        load("pb", "rank", dataset, labelling),
        fit_rank("pb", dataset, labelling),
        legend_side="right",
    )


def pb_minima(axs, dataset: str, labelling: str) -> None:
    sweep = load("pb", "rank", dataset, labelling)
    start_fms = load("pb", "start_fms", dataset, labelling)
    cp_rank.plot_minima(axs[0], cp_rank.load_minima(sweep, start_fms))


def tc2c_elbow(axs, dataset: str, labelling: str, method: str = "tc2c") -> None:
    """A Tensor-cell2cell variant's normalized error by rank: the mean over
    its random starts, banded by their range, with the Kneedle elbow
    circled."""
    ax = axs[0]
    elbow = load(method, "elbow", dataset, labelling)
    common.source_data(
        "tc2c_error_by_rank_and_run",
        elbow,
        "Normalized error (observed entries) of every Tensor-cell2cell run at "
        "every rank; plotted as the mean over runs, banded by their range.",
    )
    by_rank = elbow.groupby("Rank")["Error"]
    mean = by_rank.mean()
    band(ax, by_rank, mean)
    rank = methods.tensor_cell2cell_rank(elbow)
    if rank is not None:
        cp_rank.mark_elbow(ax, rank, mean[rank])
    cp_rank.rank_axis(ax, mean.index)
    ax.set_ylabel("Normalized error")
    title(ax, TC2C_NAMES[method], dataset, labelling, (method, "elbow"))


def tc2c_congruence(axs, dataset: str, labelling: str, method: str = "tc2c") -> None:
    """Per rank, the largest congruence product between two of a start's
    components (`rank_selection.max_congruence`): the median over starts,
    banded by their range, with the elbow rank marked. Non-negative factors
    overlap by construction, so this asks whether two components duplicate
    each other in every mode, not whether they cancel, as pseudobulk CPD's
    per-mode collinearity does."""
    ax = axs[0]
    path = COMPARATOR_RANKS_DIR / comparator_filename(
        method, "starts", dataset, labelling
    )
    with xr.open_dataset(path, engine="h5netcdf") as starts:
        congruence = methods.start_congruence(starts)
    common.source_data(
        "tc2c_max_congruence_by_rank_and_start",
        congruence,
        "Per Tensor-cell2cell rank and random start, the largest congruence "
        "product between two components: the product over the four modes of "
        "their absolute cosine (1 = two components identical up to scale).",
    )
    congruence = congruence[congruence["Rank"] > 1]
    by_rank = congruence.groupby("Rank")["MaxCongruence"]
    band(ax, by_rank, by_rank.median(), marker="o", markersize=style.MARKER_SIZE)
    common.mark_rank(ax, fit_rank(method, dataset, labelling))
    cp_rank.rank_axis(ax, np.arange(1, congruence["Rank"].max() + 1))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Max congruence product")


def tc2c_minima(axs, dataset: str, labelling: str, method: str = "tc2c") -> None:
    start_fms = load(method, "start_fms", dataset, labelling)
    sweep = tc2c_starts(dataset, labelling, method)
    cp_rank.plot_minima(axs[0], cp_rank.load_minima(sweep, start_fms))


def stac_shares(axs, dataset: str, labelling: str) -> None:
    """STACCato's eigenvalue shares per mode, the components above the 1%
    line kept (their count in the legend)."""
    ax = axs[0]
    shares = load("stac", "shares", dataset, labelling)
    common.source_data(
        "staccato_eigenvalue_shares",
        shares,
        "Each mode's eigenvalue shares of its unfolding's Gram matrix, largest "
        "first; STACCato keeps those above 1% per mode.",
    )
    core = methods.staccato_core_shape(shares)
    for color, (mode, label) in zip(
        style.CATEGORICAL, STAC_MODES.items(), strict=False
    ):
        mode_shares = shares[
            (shares["Mode"] == mode) & (shares["Component"] <= STAC_COMPONENTS)
        ]
        ax.plot(
            mode_shares["Component"],
            mode_shares["Share"],
            color=color,
            marker="o",
            markersize=style.MARKER_SIZE / 2,
            label=f"{label} ({core[mode]})",
        )
    common.reference_line(ax, methods.STACCATO_VAREXP / 100)
    ax.set_yscale("log")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue share")
    ax.legend(loc="upper right", fontsize=style.FONT_SIZE_DENSE)
    title(ax, "STAC", dataset, labelling)
