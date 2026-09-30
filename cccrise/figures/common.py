"""
This file contains functions that are used in multiple figures.
"""

import importlib
import sys
import time
from pathlib import Path
from string import ascii_letters

import anndata
import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

matplotlib.use("AGG")

matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 6
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.minor.pad"] = 0.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35
matplotlib.rcParams["svg.fonttype"] = "none"

DEFAULT_CMAP = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
DIVERGING_CMAP = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
LIGHT_DIVERGING = sns.diverging_palette(240, 10, as_cmap=True)

from ..factors import (
    ALAD_FACTORS_PATH,
    DATA_DIR,
    IBD_FACTORS_PATH,
    clean_memory,
    load_factors_to_anndata,
    load_or_compute_dataframe,
)
from ..import_data import import_alad
from ..runners.datasets import DATASETS as RUNNER_DATASETS
from ..runners.datasets import IBD_CELL_CAP

#: The IBD cohort as its factor model (`IBD_FACTORS_PATH`) was fit: capped at
#: `IBD_CELL_CAP` cells per sample. `.load()` gives those cells' expression.
IBD_SPEC = RUNNER_DATASETS["ibd"].with_cell_cap(IBD_CELL_CAP)

__all__ = [
    "ALAD_FACTORS_PATH",
    "DATA_DIR",
    "DEFAULT_CMAP",
    "DIVERGING_CMAP",
    "IBD_FACTORS_PATH",
    "IBD_SPEC",
    "LIGHT_DIVERGING",
    "clean_memory",
    "genFigure",
    "getSetup",
    "get_alad_data",
    "get_ibd_data",
    "load_factors_to_anndata",
    "load_or_compute_dataframe",
    "load_runner_csv",
    "subplotLabel",
]


def get_alad_data(
    project_cells: bool = False, do_embedding: bool = False
) -> anndata.AnnData:
    """Load ALAD dataset with factors and optional single-cell projections."""
    if not project_cells:
        return load_factors_to_anndata(ALAD_FACTORS_PATH)
    raw = import_alad(gene_threshold=0.001, normalize=True)
    return load_factors_to_anndata(
        ALAD_FACTORS_PATH, adata=raw, do_embedding=do_embedding
    )


def get_ibd_data(
    project_cells: bool = False, do_embedding: bool = False
) -> anndata.AnnData:
    """Load the IBD dataset (capped at `IBD_CELL_CAP` cells per sample, the
    cells its factor model was fit on) with factors and optional single-cell
    expression."""
    if not project_cells:
        return load_factors_to_anndata(IBD_FACTORS_PATH)
    raw = IBD_SPEC.load()
    return load_factors_to_anndata(
        IBD_FACTORS_PATH, adata=raw, do_embedding=do_embedding
    )


def load_runner_csv(filename: str) -> pd.DataFrame:
    """Read a sweep result written by one of the runner scripts.

    Figures plot what a runner already produced; they never compute a sweep
    themselves, so a missing file is an error naming the command to run.
    """
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Produce it with the matching runner, e.g.\n"
            "    uv run cccrise-riserank --dataset <dataset> --ranks 5:51:5\n"
            "    uv run cccrise-cpdrank  --dataset <dataset> --rise-rank <N>\n"
            "    uv run cccrise-anovaboot --dataset <dataset> --rise-rank <N> --cp-rank <K>"
        )
    return pd.read_csv(path)


def getSetup(
    figsize: tuple[float, float] | tuple[int, int], gridd: tuple[int, int]
) -> tuple[list[plt.Axes], Figure]:
    """Establish figure set-up with subplots."""
    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, layout="constrained")
    gs1 = gridspec.GridSpec(gridd[0], gridd[1], figure=f)

    # Get list of axis objects
    ax = [f.add_subplot(gs1[x]) for x in range(gridd[0] * gridd[1])]

    return ax, f


def subplotLabel(axs: list[plt.Axes]):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_letters[ii],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )


def genFigure():
    """Main figure generation function."""
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    module = importlib.import_module(f"cccrise.figures.{nameOut}")
    ff = module.makeFigure()

    if ff is not None:
        # Find project root (where pyproject.toml is located)
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        ff.savefig(
            output_dir / f"{nameOut}.svg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")
