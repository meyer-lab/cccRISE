"""
Functions shared across manuscript figures: data loading, panel layout, the
build entry point, and plotting functions used by more than one figure (e.g.
the same rank-selection panels for ALAD and IBD).

Each figure lives in its own `figures/figure<N>/` package: one module per
panel, each exposing `plot(fig)` (draws into a Figure or SubFigure) and
`makeFigure()` (the panel alone), plus a `figure<N>.py` master that arranges
the panels with `panel_grid` and labels them with `label_panels`.
"""

import importlib
import sys
import time
from collections.abc import Callable
from pathlib import Path
from string import ascii_letters, ascii_uppercase

import anndata
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import NullLocator

matplotlib.use("AGG")

from ..factors import (
    ALAD_FACTORS_PATH,
    COVID19_FACTORS_PATH,
    DATA_DIR,
    clean_memory,
    load_factors_to_anndata,
    load_or_compute_dataframe,
)
from ..import_data import import_alad, import_balf_covid
from ..runners.datasets import DATASETS, VALIDATION_DIR
from ..runners.provenance import read_metadata
from ..validation.component_stability import metric_interval
from . import style

__all__ = [
    "ALAD_FACTORS_PATH",
    "COVID19_FACTORS_PATH",
    "DATA_DIR",
    "clean_memory",
    "genFigure",
    "getSetup",
    "get_alad_data",
    "get_covid_data",
    "load_factors_to_anndata",
    "load_or_compute_dataframe",
    "load_runner_output",
    "subplotLabel",
]


def get_covid_data(
    project_cells: bool = False, do_embedding: bool = False
) -> anndata.AnnData:
    """Load COVID dataset with factors and optional single-cell projections."""
    if not project_cells:
        return load_factors_to_anndata(COVID19_FACTORS_PATH)
    raw = import_balf_covid(gene_threshold=0.001, normalize=True)
    return load_factors_to_anndata(
        COVID19_FACTORS_PATH, adata=raw, do_embedding=do_embedding
    )


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


def load_runner_output(
    filename: str, root: Path, hint: str | None = None
) -> pd.DataFrame:
    """Read a sweep result written by one of the runner scripts.

    Figures plot what a runner already produced; they never compute a sweep
    themselves, so a missing file is an error naming the command to run.

    Parameters
    ----------
    filename : str
        The runner's output filename.
    root : Path
        Directory the file lives in -- the runner's analysis directory
        (`runners.datasets.RANK_SELECTION_DIR` / `VALIDATION_DIR`) for a
        default-settings output, or a `runners.datasets.comparison_output_dir`
        for an alternate-score one.
    hint : str, optional
        Command to suggest in the error when the file is missing. Defaults
        to the rank-sweep runners' commands.
    """
    path = root / filename
    if not path.exists():
        hint = hint or (
            "    uv run cccrise-riserank --dataset <dataset> --ranks 1:51\n"
            "    uv run cccrise-cpdrank  --dataset <dataset> --rise-rank <N>\n"
            "    uv run cccrise-anovaboot --dataset <dataset> --rise-rank <N> --cp-rank <K>"
        )
        raise FileNotFoundError(
            f"{path} not found. Produce it with the matching runner, e.g.\n{hint}"
        )
    return pd.read_parquet(path)


def selected_ranks(dataset: str) -> tuple[int, int]:
    """(RISE rank, CP rank) of `dataset`'s canonical fit, from its provenance."""
    ranks = read_metadata(DATASETS[dataset].factors_path())["rank_selection"]
    return ranks["rise"]["rank"], ranks["cpd"]["rank"]


# =============================================================================
# Layout and build
# =============================================================================

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


def panel_grid(
    figsize: tuple[float, float], layout: list[list[str]], **gridspec_kw
) -> tuple[Figure, dict[str, SubFigure]]:
    """A figure divided into one subfigure per panel.

    `layout` is a mosaic: each row names the panel occupying each grid cell,
    and a panel named in several adjacent cells spans them. Panels are
    returned in reading order, the order `label_panels` letters them in.
    """
    fig = plt.figure(figsize=figsize, layout="constrained")
    grid = fig.add_gridspec(len(layout), len(layout[0]), **gridspec_kw)
    cells = np.array(layout)
    panels = {}
    for name in dict.fromkeys(cells.ravel()):
        rows, cols = np.nonzero(cells == name)
        panels[name] = fig.add_subfigure(
            grid[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        )
    return fig, panels


def label_panels(panels: dict[str, SubFigure]) -> None:
    """Label panels A, B, C, ... in order, at each panel's top left."""
    for letter, panel in zip(ascii_uppercase, panels.values(), strict=False):
        panel.suptitle(letter, x=0, ha="left")


def compose(
    figsize: tuple[float, float], layout: list[list[str]], panels: dict, **gridspec_kw
) -> Figure:
    """A master figure: each panel module in `panels` (name -> module with a
    `plot(fig)`) drawn into its `layout` cells, then lettered."""
    fig, subfigs = panel_grid(figsize, layout, **gridspec_kw)
    for name, subfig in subfigs.items():
        panels[name].plot(subfig)
    label_panels(subfigs)
    return fig


def standalone(
    plot: Callable[[Figure | SubFigure], None],
    size: tuple[float, float],
    legend: Callable[[Figure], None] | None = None,
) -> Figure:
    """One panel drawn on a figure of its own, with `legend(fig)` for panels
    whose master figure shares one legend across panels."""
    fig = plt.figure(figsize=size, layout="constrained")
    plot(fig)
    if legend is not None:
        legend(fig)
    return fig


def placeholder(fig: Figure | SubFigure, text: str) -> None:
    """A dashed box marking a panel that is not implemented yet."""
    ax = fig.subplots()
    ax.set_axis_off()
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            linestyle=style.LINE_STYLE_REFERENCE,
            edgecolor=style.COLORS["muted"],
        )
    )
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=style.COLORS["reference"],
    )


def build_figure(figure: str, panel: str | None = None) -> Figure:
    """Figure `figure` (e.g. '2', 'S_stability'), or one of its panels."""
    style.apply_style()
    package = f"figure{figure}"
    module = importlib.import_module(f"cccrise.figures.{package}.{panel or package}")
    return module.makeFigure()


def genFigure():
    """Build a figure, or one of its panels, into output/figure<N>/:

    uv run cccrise-fbuild 2               # figures/figure2/figure2.py
    uv run cccrise-fbuild 2 rise_rank     # figures/figure2/rise_rank.py
    uv run cccrise-fbuild 2 --png         # PNG instead of SVG
    """
    start = time.time()
    args = [arg for arg in sys.argv[1:] if arg != "--png"]
    extension = "png" if "--png" in sys.argv else "svg"
    figure, panel = args[0], (args[1] if len(args) > 1 else None)

    fig = build_figure(figure, panel)
    output_dir = OUTPUT_DIR / f"figure{figure}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": figures are drawn at their final print size.
    fig.savefig(output_dir / f"{panel or f'figure{figure}'}.{extension}")
    print(f"Figure {' '.join(args)} is done after {time.time() - start:.1f} s.\n")


# =============================================================================
# Shared panels
# =============================================================================

SIG_ALPHA = 0.05


def mark_rank(ax: plt.Axes, rank: int | None) -> None:
    """Vertical reference line at the rank the canonical fit uses."""
    if rank is not None:
        ax.axvline(
            rank,
            color=style.COLORS["reference"],
            linestyle=style.LINE_STYLE_REFERENCE,
            linewidth=style.LINE_WIDTH_THIN,
            zorder=0,
        )


def reference_line(
    ax: plt.Axes,
    value: float,
    horizontal: bool = True,
    linestyle: str = style.LINE_STYLE_REFERENCE,
    label: str | None = None,
):
    """Null, chance, or threshold line in the shared reference style."""
    return (ax.axhline if horizontal else ax.axvline)(
        value,
        color=style.COLORS["reference"],
        linestyle=linestyle,
        linewidth=style.LINE_WIDTH_THIN,
        label=label,
    )


def significance_legend(ax: plt.Axes, alpha: float = SIG_ALPHA, **kwargs) -> None:
    """One-row legend for the up / down / not-significant coloring."""
    handles = [
        Patch(color=style.COLORS["up"], label="Up"),
        Patch(color=style.COLORS["down"], label="Down"),
        Patch(color=style.COLORS["not_significant"], label="n.s."),
    ]
    ax.legend(handles=handles, title=f"p < {alpha}", ncols=3, **kwargs)


def significance_colors(
    effect: np.ndarray, p_values: np.ndarray, center: float = 0.0, alpha=SIG_ALPHA
) -> np.ndarray:
    """Up / down / not-significant color per entry, by sign of `effect - center`."""
    direction = np.where(effect > center, style.COLORS["up"], style.COLORS["down"])
    significant = np.nan_to_num(p_values, nan=1.0) < alpha
    return np.where(significant, direction, style.COLORS["not_significant"])


# --- Rank selection (ALAD Fig 2, IBD Figs 4-5) --------------------------------


def plot_rise_rank(sweep: pd.DataFrame, ax: plt.Axes, rank: int | None = None):
    """In-sample fit R2X and held-out BiCV R2X (mean +/- SD over repeats)
    against RISE rank, from `cccrise-riserank`. BiCV peaks or plateaus near
    the rank that generalizes to unseen cells and genes."""
    for metric, color in (
        ("Fit R2X", style.COLORS["fit"]),
        ("BiCV R2X", style.COLORS["heldout"]),
    ):
        stats = sweep[sweep["Metric"] == metric].groupby("Rank")["R2X"]
        mean, sd = stats.mean(), stats.std().fillna(0)
        ax.plot(mean.index, mean, marker="o", color=color, label=metric[:-4])
        ax.fill_between(
            mean.index, mean - sd, mean + sd, color=color, alpha=0.25, linewidth=0
        )
    mark_rank(ax, rank)
    ax.set_xlabel("RISE rank")
    ax.set_ylabel("$R^2X$")
    ax.legend()


def plot_cpd_fit_r2x(sweep: pd.DataFrame, ax: plt.Axes, rank: int | None = None):
    """Full-tensor CP fit R2X against CP rank, from `cccrise-cpdrank`."""
    ax.plot(sweep["Rank"], sweep["FitR2X"], marker="o", color=style.COLORS["fit"])
    mark_rank(ax, rank)
    ax.set_xlabel("CP rank")
    ax.set_ylabel("Fit $R^2X$")


def plot_collinearity(sweep: pd.DataFrame, ax: plt.Axes, rank: int | None = None):
    """Number of modes the most collinear component pair is collinear in, per
    CP rank, marked by whether that pair duplicates or cancels itself. A
    canceling pair (collinear in >= 2 modes with opposing signs) flags a
    degenerate fit."""
    ax.plot(
        sweep["Rank"],
        sweep["MaxModesCollinear"],
        color=style.COLORS["muted"],
        zorder=1,
    )
    # NaN = no collinear pair; `fillna` keeps those rows out of both masks.
    flag = sweep["WorstPairCanceling"]
    none_found = flag.isna()
    canceling = flag.fillna(False).astype(bool) & ~none_found
    for mask, color, marker, label in (
        (none_found, style.COLORS["muted"], "o", "None collinear"),
        (~canceling & ~none_found, style.COLORS["emphasis"], "o", "Near-duplicate"),
        (canceling, style.COLORS["up"], "D", "Canceling"),
    ):
        ax.scatter(
            sweep.loc[mask, "Rank"],
            sweep.loc[mask, "MaxModesCollinear"],
            color=color,
            marker=marker,
            s=style.POINT_SIZE,
            label=label,
            zorder=2,
        )
    reference_line(ax, 2)
    mark_rank(ax, rank)
    ax.set_ylim(-0.3, 7.5)  # headroom for the legend
    ax.set_yticks(range(5))
    ax.set_xlabel("CP rank")
    ax.set_ylabel("Modes collinear (worst pair)")
    ax.legend(loc="upper left", fontsize=style.FONT_SIZE_DENSE)


def plot_core_consistency(sweep: pd.DataFrame, ax: plt.Axes, rank: int | None = None):
    """CORCONDIA core consistency against CP rank, with the 90% and 50%
    conventional thresholds."""
    ax.plot(
        sweep["Rank"], sweep["CoreConsistency"], marker="o", color=style.COLORS["fit"]
    )
    reference_line(ax, 90)
    reference_line(ax, 50, linestyle=":")
    mark_rank(ax, rank)
    # Core consistency collapses by orders of magnitude past the valid-model
    # rank; symlog keeps the 0-100% range readable while showing the collapse,
    # with a handful of evenly spaced negative decades as ticks.
    ax.set_yscale("symlog", linthresh=100)
    lowest = int(np.ceil(np.log10(max(-sweep["CoreConsistency"].min(), 100))))
    step = max(1, int(np.ceil((lowest - 2) / 4)))
    ax.set_yticks([100] + [-(10.0**k) for k in range(2 + step, lowest + 1, step)])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(top=150)
    ax.set_xlabel("CP rank")
    ax.set_ylabel("Core consistency (%)")


# --- Permutation significance (ALAD Fig 2, IBD Fig 5) ---------------------------


def plot_significance_calibration(
    lr_table: pd.DataFrame, ax: plt.Axes, cohort_rates: dict[str, float] | None = None
):
    """Distribution over LR pairs of the fraction of condition x eigenstate
    entries significant at p < 0.05, against the 5% expected under the null.
    `cohort_rates` maps a label (e.g. 'p < 0.05') to its cohort-wide rate."""
    fractions = lr_table["FractionSignificant_p05"].dropna()
    x_max = max(0.25, np.ceil(fractions.max() * 20) / 20)
    sns.histplot(
        fractions,
        kde=True,
        ax=ax,
        color=style.COLORS["emphasis"],
        bins=np.linspace(0, x_max, 26),
        stat="density",
        alpha=0.4,
        edgecolor="white",
        linewidth=style.LINE_WIDTH_THIN,
    )
    reference_line(ax, SIG_ALPHA, horizontal=False, label="Null (5%)")
    if cohort_rates:
        ax.text(
            0.97,
            0.75,
            "\n".join(f"{k}: {v:.1%}" for k, v in cohort_rates.items()),
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Fraction of entries significant\nper LR pair (p < 0.05)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")


def plot_observed_vs_null(lr_table: pd.DataFrame, ax: plt.Axes):
    """Mean observed communication score against the permutation-null mean,
    per LR pair, colored by mean z-score."""
    observed, null = lr_table["MeanObserved"], lr_table["MeanNull"]
    z = lr_table["MeanZScore"]
    limit = np.abs(z).max()
    points = ax.scatter(
        observed,
        null,
        c=z,
        cmap=style.CMAPS["diverging"],
        vmin=-limit,
        vmax=limit,
        s=style.POINT_SIZE_BACKGROUND,
        edgecolors="none",
        rasterized=True,
    )
    ax.figure.colorbar(points, ax=ax, label="Mean z-score", shrink=0.8)
    low = min(0.0, observed.min(), null.min())
    high = max(observed.max(), null.max()) * 1.05
    ax.plot(
        [low, high],
        [low, high],
        color=style.COLORS["reference"],
        linestyle=style.LINE_STYLE_REFERENCE,
        linewidth=style.LINE_WIDTH_THIN,
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Mean observed score")
    ax.set_ylabel("Mean permutation-null score")


def plot_differential_volcano(
    differential: pd.DataFrame,
    ax: plt.Axes,
    contrast: str = "Group 1 - Group 0",
    clip_quantile: float = 0.998,
):
    """Cross-group communication difference against -log10 p, per sender x
    receiver eigenstate x LR pair. Differences beyond the `clip_quantile`
    magnitude are drawn as arrowheads at the axis edge."""
    diff = differential["ObservedDiff"].to_numpy()
    p_values = differential["PValue"].to_numpy()
    neg_log_p = -np.log10(np.maximum(p_values, 1e-6))
    limit = max(np.quantile(np.abs(diff), clip_quantile) * 1.15, 1e-12)
    colors = significance_colors(diff, p_values)
    clipped = np.abs(diff) > limit
    x = np.clip(diff, -limit, limit)

    for mask, marker, size in (
        (~clipped, "o", style.POINT_SIZE_BACKGROUND),
        (clipped & (diff > 0), ">", style.POINT_SIZE),
        (clipped & (diff < 0), "<", style.POINT_SIZE),
    ):
        ax.scatter(
            x[mask],
            neg_log_p[mask],
            c=colors[mask],
            marker=marker,
            s=size,
            edgecolors="none",
            rasterized=True,
        )
    reference_line(ax, -np.log10(SIG_ALPHA))
    ax.set_xlim(-limit * 1.05, limit * 1.05)
    ax.set_ylim(0, neg_log_p.max() * 1.7)  # headroom for the legend
    ax.set_xlabel(f"Communication difference\n({contrast})")
    ax.set_ylabel(r"$-\log_{10}\ p$")
    significance_legend(ax, loc="upper center")


# --- NicheNet downstream-target validation (ALAD Fig 2, IBD Fig 5) -------------

NICHENET_MODES = {"within_celltype": "Within cell type", "vs_rest": "Vs. rest"}


def nichenet_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Per receiver mode and component: mean ligand AUROC, ligands scored,
    and ligands significant against the permutation null."""
    return scores.groupby(["receiver_mode", "component"]).agg(
        mean_auroc=("auroc", "mean"),
        n_scored=("auroc", "count"),
        n_significant=("p_value", lambda p: int((p < SIG_ALPHA).sum())),
    )


def plot_nichenet_summary(summary: pd.DataFrame, ax: plt.Axes, value: str, ylabel: str):
    """One bar per component and receiver mode for `value`, a column of
    `nichenet_summary`; a chance line is drawn for the AUROC."""
    table = summary[value].unstack("receiver_mode")[list(NICHENET_MODES)]
    width = 0.8 / table.shape[1]
    x = np.arange(len(table))
    for i, (mode, label) in enumerate(NICHENET_MODES.items()):
        ax.bar(
            x + (i - (table.shape[1] - 1) / 2) * width,
            table[mode],
            width,
            color=style.COLORS[mode],
            label=label,
        )
    if value == "mean_auroc":
        reference_line(ax, 0.5)
        ax.set_ylim(0, 1)
    ax.set_xticks(x, table.index)
    ax.set_xlabel("CP component")
    ax.set_ylabel(ylabel)
    ax.legend(title="Receivers")


def top_nichenet_ligands(scores: pd.DataFrame, n_top: int = 10) -> pd.DataFrame:
    """Each component's `n_top` highest-loading ligands, one row per ligand (a
    ligand's AUROC does not depend on its receptor), with a within-component
    'rank'. Ligands absent from the NicheNet database keep a NaN AUROC."""
    by_ligand = scores.sort_values("loading", key=np.abs, ascending=False)
    by_ligand = by_ligand.drop_duplicates(["component", "ligand"])
    top = by_ligand.groupby("component").head(n_top).copy()
    top["rank"] = top.groupby("component").cumcount()
    return top


def plot_nichenet_heatmap(scores: pd.DataFrame, ax: plt.Axes, n_top: int = 10):
    """Ligand-activity AUROC of each component's top-loading ligands: one
    column per component, rows by loading rank, each cell labeled with its
    ligand. Significant ligands (permutation null) are bold with a star;
    ligands absent from the NicheNet database are grey."""
    top = top_nichenet_ligands(scores, n_top)
    auroc = top.pivot(index="rank", columns="component", values="auroc")
    limit = np.nanmax(np.abs(auroc.to_numpy() - 0.5))
    image = ax.imshow(
        auroc.to_numpy(),
        cmap=style.CMAPS["diverging"],
        vmin=0.5 - limit,
        vmax=0.5 + limit,
        aspect="auto",
    )
    columns = {component: i for i, component in enumerate(auroc.columns)}
    for component, rank, ligand, value, p_value in zip(
        top["component"],
        top["rank"],
        top["ligand"],
        top["auroc"],
        top["p_value"],
        strict=True,
    ):
        x = columns[component]
        if np.isnan(value):
            ax.add_patch(
                Rectangle(
                    (x - 0.5, rank - 0.5), 1, 1, color=style.COLORS["not_significant"]
                )
            )
        significant = p_value < SIG_ALPHA
        ax.text(
            x,
            rank,
            ligand + ("*" if significant else ""),
            ha="center",
            va="center",
            # Complex ligands (e.g. ITGA4&ITGB7) would overflow their cell.
            fontsize=style.FONT_SIZE_DENSE - (len(ligand) > 9),
            fontweight="bold" if significant else "normal",
        )
    ax.set_xticks(range(auroc.shape[1]), auroc.columns)
    ax.set_yticks([])
    ax.set_xlabel("CP component")
    ax.set_ylabel(f"Top {n_top} ligands by loading")
    ax.tick_params(axis="x", length=0)
    ax.spines[:].set_visible(False)
    ax.figure.colorbar(image, ax=ax, label="Ligand AUROC", shrink=0.8)


# --- Component stability (ALAD Fig 2 and supplement) -------------------------------------------


def load_stability(dataset: str) -> pd.DataFrame:
    """Replicate-level component stability table from `cccrise-stability`."""
    return load_runner_output(DATASETS[dataset].stability_filename(), VALIDATION_DIR)


def plot_per_component_fms(stability: pd.DataFrame, ax: plt.Axes, stage: str):
    """Per-component FMS over resampling replicates, one box per axis, for
    `stage` ('RISE' or 'CPD'). The grey seed-only box is the reproducibility
    ceiling for that component: the gap below it is what the data
    perturbation costs beyond reseeding."""
    rows = stability[stability["Stage"] == stage]
    axes = [a for a in style.STABILITY_AXIS_COLORS if a in set(rows["Axis"])]
    sns.boxplot(
        data=rows[rows["Axis"].isin(axes)],
        x="Component",
        y="FMS",
        hue="Axis",
        hue_order=axes,
        palette=style.STABILITY_AXIS_COLORS,
        fliersize=1,
        linewidth=style.LINE_WIDTH_THIN,
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"{stage} component")
    ax.set_ylabel("Factor match score")


def stability_legend(fig: Figure | SubFigure) -> None:
    """One legend of the resampling-axis colors, above `fig`."""
    handles = [
        Patch(color=color, label=axis)
        for axis, color in style.STABILITY_AXIS_COLORS.items()
    ]
    fig.legend(handles=handles, title="Resampled", loc="outside upper center", ncols=4)


def plot_aggregate_fms(stability: pd.DataFrame, ax: plt.Axes):
    """Aggregate FMS (a replicate's mean over components) per stage and
    axis: median with a 2.5-97.5 percentile band over replicates."""
    stats = metric_interval(
        stability[stability["Axis"].isin(style.STABILITY_AXIS_COLORS)],
        "FMS",
        by=("Stage", "Axis"),
        lower=2.5,
        upper=97.5,
    )
    x = np.arange(len(stats))
    ax.errorbar(
        x,
        stats["Median"],
        yerr=[stats["Median"] - stats["P2.5"], stats["P97.5"] - stats["Median"]],
        fmt="none",
        ecolor=style.COLORS["ink"],
        capsize=2,
        linewidth=style.LINE_WIDTH_THIN,
    )
    ax.scatter(
        x,
        stats["Median"],
        c=[style.STABILITY_AXIS_COLORS[a] for a in stats["Axis"]],
        s=style.POINT_SIZE_FOREGROUND,
        zorder=3,
    )
    ax.set_xticks(
        x,
        [f"{s}: {a}" for s, a in zip(stats["Stage"], stats["Axis"], strict=True)],
        rotation=30,
        ha="right",
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Aggregate FMS")


# =============================================================================
# Legacy layout helpers, used only by figures/legacy (including
# legacy/commonFuncs); removed with them.
# =============================================================================


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
