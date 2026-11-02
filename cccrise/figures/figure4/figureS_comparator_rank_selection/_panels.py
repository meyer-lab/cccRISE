"""Each annotation-pooled comparator's own rank selection on one labelling,
from ``uv run cccrise-comparators``: pseudobulk CPD by cccRISE's seeded-start
sweep (Figure 2's panels: fit R2X across starts, the LR pair mode's
collinearity per start, the distinct minima), Tensor-cell2cell by the Kneedle
elbow of its error (with how nearly two of a start's components duplicate
each other, per start, and the minima of its random starts), and STACCato by
each mode's eigenvalue shares above 1%."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ....comparators import methods
from ....rank_selection import distinct_minima
from ....runners.datasets import (
    COMPARATOR_FITS_DIR,
    COMPARATOR_RANKS_DIR,
    comparator_filename,
)
from ... import common, style

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
#: Where the per-start key sits (coarse pseudobulk CPD's collinearity only;
#: the other per-start panels share its colors).
KEY_LOC = "upper left"


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


def minima(method: str, sweep: str, dataset: str, labelling: str) -> pd.DataFrame:
    """`rank_selection.distinct_minima` of `method`'s seeded starts (its
    `sweep` table for each start's R2X)."""
    starts = load(method, sweep, dataset, labelling)
    return distinct_minima(
        load(method, "start_fms", dataset, labelling),
        starts.set_index(["Rank", "Seed"])["FitR2X"],
    )


def title(ax: plt.Axes, name: str, labelling: str) -> None:
    ax.set_title(f"{name} {labelling}", loc="left")


def rank_ticks(ax: plt.Axes) -> None:
    """Thin a `common.rank_axis` axis's ticks: every rank up to 10, every
    second up to 17, then every fourth (odd ranks crowd a square axis past
    17)."""
    max_rank = round(ax.get_xlim()[1] - 0.6)
    step = 1 if max_rank <= 10 else 2 if max_rank <= 17 else 4
    ax.set_xticks(np.arange(1, max_rank + 1, step))


def band(ax: plt.Axes, by_rank, center: pd.Series, **kwargs) -> None:
    """`center` by rank, banded by `by_rank`'s range."""
    color = style.COLORS["fit"]
    ax.fill_between(
        center.index, by_rank.min(), by_rank.max(), color=color, alpha=0.2, lw=0
    )
    ax.plot(center.index, center, color=color, **kwargs)


def pb_fit(axs, dataset: str, labelling: str) -> None:
    common.plot_r2x_sweep(
        axs[0],
        load("pb", "rank", dataset, labelling),
        fit_rank("pb", dataset, labelling),
    )
    rank_ticks(axs[0])
    title(axs[0], "PB", labelling)


def pb_collinearity(axs, dataset: str, labelling: str) -> None:
    """The LR pair mode's largest |cosine| between two components per start,
    as Figure 2C."""
    sweep = load("pb", "rank", dataset, labelling)
    common.source_data(
        "pb_lr_max_cosine_by_rank_and_start",
        sweep[["Rank", "Start", "Seed", "MaxCosD"]],
        "Per pseudobulk CPD rank and seeded SVD start, the largest absolute "
        "cosine between two components' ligand-receptor pair factors.",
    )
    common.plot_start_boxes(
        axs[0],
        sweep[sweep["Rank"] > 1],
        "MaxCosD",
        minima("pb", "rank", dataset, labelling),
        fit_rank("pb", dataset, labelling),
        wrap_key=True,
        key_loc=KEY_LOC if labelling == "coarse" else None,
    )
    rank_ticks(axs[0])
    axs[0].set_ylabel("LR pair max |cosine|")


def pb_minima(axs, dataset: str, labelling: str) -> None:
    common.plot_minima_table(
        axs[0],
        minima("pb", "rank", dataset, labelling),
        fit_rank("pb", dataset, labelling),
    )
    rank_ticks(axs[0])


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
        common.mark_kneedle(ax, rank, mean[rank], above=True)
    common.mark_rank(ax, fit_rank(method, dataset, labelling))
    common.rank_axis(ax, mean.index)
    rank_ticks(ax)
    ax.set_ylabel("Normalized error")
    title(ax, TC2C_NAMES[method], labelling)


def tc2c_congruence(axs, dataset: str, labelling: str, method: str = "tc2c") -> None:
    """Per rank, the largest congruence product between two of a start's
    components (`rank_selection.max_congruence`), per start as Figure 2C's
    collinearity, with the elbow rank marked. Non-negative factors overlap
    by construction, so this asks whether two components duplicate each
    other in every mode, not whether they cancel, as pseudobulk CPD's
    collinearity does."""
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
    common.plot_start_boxes(
        ax,
        congruence[congruence["Rank"] > 1],
        "MaxCongruence",
        minima(method, "elbow", dataset, labelling),
        fit_rank(method, dataset, labelling),
        key_loc=None,
    )
    rank_ticks(ax)
    ax.set_ylabel("Max congruence product")


def tc2c_minima(axs, dataset: str, labelling: str, method: str = "tc2c") -> None:
    common.plot_minima_table(
        axs[0],
        minima(method, "elbow", dataset, labelling),
        fit_rank(method, dataset, labelling),
    )
    rank_ticks(axs[0])


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
    title(ax, "STAC", labelling)
