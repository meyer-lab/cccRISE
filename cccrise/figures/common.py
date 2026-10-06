"""
Functions shared across manuscript figures: data loading, panel layout, the
build entry point, and plotting functions used by more than one figure (e.g.
the same rank-selection panels for ALAD and IBD). Code used by a single panel
lives in that panel's module instead, and settings, data, or functions shared
only by one figure's panels live in that figure's private `_common.py`.

Each figure lives in its own `figures/figure<N>/` package: one module per
panel, each exposing `plot(fig)` (draws into a Figure or SubFigure) and
`makeFigure()` (the panel alone), plus a `figure<N>.py` master that arranges
the panels with `panel_grid` and labels them with `label_panels`.
"""

import functools
import importlib
import sys
import time
from collections.abc import Callable, Sequence
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

from ..anova import (
    get_sample_to_group,
    prepare_group_labels,
)
from ..factors import (
    ALAD_FACTORS_PATH,
    COVID19_FACTORS_PATH,
    DATA_DIR,
    clean_memory,
    load_factors_to_anndata,
    load_or_compute_dataframe,
)
from ..import_data import import_alad, import_balf_covid
from ..runners.datasets import DATASETS
from ..runners.provenance import read_metadata
from ..utils import (
    celltype_component_association,
    correct_conditions,
)
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
            "    uv run cccrise-stability --dataset <dataset> --rise-rank <N> --cp-rank <K>"
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


def plot_cpd_fit_r2x(
    sweep: pd.DataFrame,
    ax: plt.Axes,
    rank: int | None = None,
    color: str = style.COLORS["fit"],
):
    """Full-tensor CP fit R2X against CP rank, from `cccrise-cpdrank`."""
    ax.plot(sweep["Rank"], sweep["FitR2X"], marker="o", color=color)
    mark_rank(ax, rank)
    ax.set_xlabel("CP rank")
    ax.set_ylabel("Fit $R^2X$")


def plot_collinearity(
    sweep: pd.DataFrame,
    ax: plt.Axes,
    rank: int | None = None,
    colors: tuple[str, str, str] = (
        style.COLORS["muted"],
        style.COLORS["emphasis"],
        style.COLORS["up"],
    ),
):
    """Number of modes the most collinear component pair is collinear in, per
    CP rank, marked by whether that pair duplicates or cancels itself. A
    canceling pair (collinear in >= 2 modes with opposing signs) flags a
    degenerate fit. `colors` is (no collinear pair, near-duplicate,
    canceling)."""
    none_color, duplicate_color, canceling_color = colors
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
        (none_found, none_color, "o", "None collinear"),
        (~canceling & ~none_found, duplicate_color, "o", "Near-duplicate"),
        (canceling, canceling_color, "D", "Canceling"),
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


# --- Outcome associations (ALAD Fig 3, IBD Fig 5) ------------------------------

#: Outcome levels in reading order (healthy/control first); unlisted levels
#: follow alphabetically.
LEVEL_ORDER = [
    "control",
    "declined",
    "recovered",
    "No CLAD/Stable",
    "No CLAD/Recovered",
    "No CLAD",
    "CLAD",
    "CLAD/Declined",
    "CLAD/Stable",
    "CLAD/Died",
    "Died",
    "no",
    "yes",
    "stable",
    "improved",
    "ILD",
    "Other",
    "IgG-",
    "IgG+",
    "female",
    "male",
]


def plot_association_heatmap(table: pd.DataFrame, ax: plt.Axes) -> None:
    """Eta-squared of each CP component (columns) against each outcome
    association (rows, in `table` order), from `anova.run_multiple_factors_anova`;
    bold with a star where the FDR-adjusted p is below `SIG_ALPHA`."""
    order = list(dict.fromkeys(table["Association"]))
    effect, padj = (
        table.pivot(index="Association", columns="Component", values=v).reindex(order)
        for v in ("EtaSquared", "PAdj")
    )
    values = effect.to_numpy(dtype=float)
    vmax = np.nanmax(values)
    image = ax.imshow(
        values, cmap=style.CMAPS["sequential"], vmin=0, vmax=vmax, aspect="auto"
    )
    for (i, j), value in np.ndenumerate(values):
        if not np.isfinite(value):
            continue
        significant = padj.iloc[i, j] < SIG_ALPHA
        ax.text(
            j,
            i,
            f"{value:.2f}" + ("*" if significant else ""),
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold" if significant else "normal",
            color="white" if value > 0.6 * vmax else style.COLORS["ink"],
        )
    ax.set_xticks(range(effect.shape[1]), effect.columns)
    ax.set_yticks(range(len(order)), order, fontsize=style.FONT_SIZE_DENSE)
    ax.set_xlabel("CP component")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.figure.colorbar(image, ax=ax, label=r"$\eta^2$", shrink=0.8)


def plot_association_effect_sizes(
    table: pd.DataFrame,
    ax: plt.Axes,
    highlight_component: int | None = None,
    emphasis_color: str = style.COLORS["emphasis"],
    muted_color: str = style.COLORS["muted"],
) -> None:
    """Eta-squared per CP component for one association, starred where the
    adjusted p is below `SIG_ALPHA`. Significant bars are emphasized, or only
    `highlight_component` when given."""
    rows = table.sort_values("Component")
    components = rows["Component"].astype(int).to_numpy()
    p_values = rows["PAdj" if "PAdj" in rows else "P"].to_numpy(dtype=float)
    significant = p_values < SIG_ALPHA
    emphasized = (
        significant
        if highlight_component is None
        else components == highlight_component
    )
    values = rows["EtaSquared"].to_numpy(dtype=float)
    x = np.arange(len(components))
    ax.bar(x, values, color=np.where(emphasized, emphasis_color, muted_color))
    for xi, value in zip(x[significant], values[significant], strict=True):
        ax.text(xi, value, "*", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x, components)
    ax.set_xlabel("CP component")
    ax.set_ylabel(r"$\eta^2$")
    ax.set_ylim(0, np.nanmax(values) * 1.2)
    if "Association" in rows:
        ax.set_title(str(rows["Association"].iloc[0]))


def group_loadings(
    adata: anndata.AnnData,
    component: int,
    group_column: str,
    condition_column: str,
    **regrouping,
) -> pd.DataFrame:
    """Read-depth-corrected loading of one CP component per condition, with
    its (regrouped, see `anova.prepare_group_labels`) outcome level; conditions
    without a level are dropped."""
    labels = prepare_group_labels(
        get_sample_to_group(adata, condition_column, group_column), **regrouping
    )
    return pd.DataFrame(
        {
            "Condition": labels.index,
            "Status": labels.to_numpy(),
            "Loading": correct_conditions(adata)[:, component - 1],
        }
    ).dropna(subset=["Status"])


def _ordered_levels(levels) -> list:
    rank = {level: i for i, level in enumerate(LEVEL_ORDER)}
    return sorted(set(levels), key=lambda level: (rank.get(level, len(rank)), level))


def _join_patients(ax, points, adata, condition_column, patient_column) -> None:
    """Line through the points of conditions that share a patient."""
    patient = get_sample_to_group(adata, condition_column, patient_column)
    points = points.assign(Patient=patient.loc[points["Condition"]].to_numpy())
    for _, samples in points.groupby("Patient"):
        if len(samples) > 1:
            ax.plot(
                samples["x"],
                samples["Loading"],
                color=style.COLORS["reference"],
                linewidth=style.LINE_WIDTH_THIN,
                zorder=2,
            )


def plot_category_component_weights(
    adata: anndata.AnnData,
    component: int,
    group_column: str,
    ax: plt.Axes,
    title: str | None = None,
    anova_df: pd.DataFrame | None = None,
    patient_column: str | None = None,
    condition_column: str = "dsco_id",
    palette: Sequence[str] = style.CATEGORICAL,
    **regrouping,
) -> None:
    """Box plot of one CP component's condition loadings per outcome level,
    with the conditions as jittered points, joined per patient when
    `patient_column` is given. `regrouping` (mapping / levels /
    control_label / other_label) is passed to `anova.prepare_group_labels`;
    `anova_df` adds the component's F, adjusted p, and eta-squared to the
    title."""
    points = group_loadings(
        adata, component, group_column, condition_column, **regrouping
    )
    levels = _ordered_levels(points["Status"])
    counts = points["Status"].value_counts()
    sns.boxplot(
        data=points,
        x="Status",
        y="Loading",
        order=levels,
        hue="Status",
        hue_order=levels,
        palette=list(palette[: len(levels)]),
        legend=False,
        width=0.5,
        fliersize=0,
        linewidth=style.LINE_WIDTH_THIN,
        boxprops={"alpha": 0.8},
        ax=ax,
    )
    # Jittered by hand so patient lines can run through the exact points.
    rng = np.random.default_rng(0)
    points["x"] = points["Status"].map({level: i for i, level in enumerate(levels)})
    points["x"] += rng.uniform(-0.2, 0.2, len(points))
    ax.scatter(
        points["x"],
        points["Loading"],
        color=style.COLORS["ink"],
        s=style.POINT_SIZE,
        zorder=3,
    )
    if patient_column is not None:
        _join_patients(ax, points, adata, condition_column, patient_column)

    # Wrapped at "/" rather than rotated, so long levels stay legible.
    ax.set_xticks(
        range(len(levels)),
        [f"{str(level).replace('/', '/\n')}\n(n={counts[level]})" for level in levels],
    )
    ax.set_xlabel("")
    ax.set_ylabel(f"Component {component} loading")
    ax.set_title(title or group_column)
    ax.text(
        0.03,
        0.97,
        _anova_stats(anova_df, component),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style.FONT_SIZE_DENSE,
    )


def _anova_stats(anova_df: pd.DataFrame | None, component: int) -> str:
    """Title suffix with one component's F, adjusted p, and eta-squared."""
    if anova_df is None:
        return ""
    row = anova_df[anova_df["Component"] == component]
    if row.empty:
        return ""
    f, padj, eta = (float(row[c].iloc[0]) for c in ("F", "PAdj", "EtaSquared"))
    return f"$F$ = {f:.2f}\n$p_{{adj}}$ = {padj:.2g}\n$\\eta^2$ = {eta:.2f}"


# =============================================================================
# Cell type granularity: eigenstate and per-cell sender/receiver weights
# =============================================================================

STRONG_CELL_PERCENTILE = 90  # Shiny explorer's default Strength Threshold
N_LABELED_EIGENSTATES = 3


@functools.cache
def get_cell_projections(dataset: str) -> anndata.AnnData:
    """`dataset`'s factors with per-cell eigenstate projections
    (`obsm["projections"]`) and sender/receiver weights (`obsm["sc_B"]`,
    `obsm["rc_C"]`), cached so the panels of one figure project once."""
    spec = DATASETS[dataset]
    return load_factors_to_anndata(spec.factors_path(), adata=spec.load())


def plot_celltype_eigenstate_association(
    adata: anndata.AnnData,
    dataset: str,
    ax: plt.Axes,
    highlight: Sequence[int] = (),
) -> None:
    """Cell type (rows, `style.CELL_TYPES` order) x RISE eigenstate effect
    size (Cohen's d of a cell type's projections against all other cells),
    with the `highlight` eigenstates' labels bold."""
    table = celltype_component_association(
        adata,
        factor_type="eigenstate",
        celltype_col=DATASETS[dataset].celltype_column,
        stat="effect_size",
    ).reindex(style.CELL_TYPES[dataset])
    vmax = np.nanmax(np.abs(table.to_numpy()))
    image = ax.imshow(
        table.to_numpy(),
        cmap=style.CMAPS["diverging"],
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(len(table)), table.index)
    for label in ax.get_xticklabels():
        if int(label.get_text()) in highlight:
            label.set_fontweight("bold")
    ax.set_xlabel("RISE eigenstate")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.figure.colorbar(image, ax=ax, label="Cohen's d (vs. other cells)", shrink=0.8)


def plot_component_eigenstate_loadings(
    adata: anndata.AnnData, component: int, ax: plt.Axes
) -> None:
    """A CP component's sender (B) and receiver (C) loading on each RISE
    eigenstate, labeling the `N_LABELED_EIGENSTATES` largest loadings."""
    sender, receiver = (np.asarray(adata.uns[k])[:, component - 1] for k in "BC")
    eigenstates = np.arange(1, len(sender) + 1)
    width = 0.4
    for offset, loading, role in ((-1, sender, "sender"), (1, receiver, "receiver")):
        ax.bar(
            eigenstates + offset * width / 2,
            loading,
            width,
            color=style.COLORS[role],
            label=f"{role.capitalize()} ({'B' if role == 'sender' else 'C'})",
        )
    ax.axhline(0, color=style.COLORS["reference"], linewidth=style.LINE_WIDTH_THIN)
    largest = np.abs(np.column_stack([sender, receiver])).max(axis=1)
    labeled = np.sort(eigenstates[np.argsort(-largest)[:N_LABELED_EIGENSTATES]])
    ax.set_xticks(labeled, labeled, fontweight="bold")
    ax.set(xlabel="RISE eigenstate", ylabel="Loading")
    ax.legend(loc="upper left")


def plot_celltype_scatter(
    adata: anndata.AnnData,
    dataset: str,
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    highlight: str | None = None,
) -> None:
    """Per-cell `x` vs. `y` colored by cell type, the most abundant types
    drawn beneath, the `highlight` cell type enlarged on top, and dashed lines
    at each axis's `STRONG_CELL_PERCENTILE`."""
    celltypes = adata.obs[DATASETS[dataset].celltype_column].astype(str).to_numpy()
    colors = style.CELL_TYPE_COLORS[dataset]
    counts = pd.Series(celltypes).value_counts()
    order = np.argsort(-counts[celltypes].to_numpy(), kind="stable")
    ax.scatter(
        x[order],
        y[order],
        c=[colors[c] for c in celltypes[order]],
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    if highlight is not None:
        is_highlight = celltypes == highlight
        ax.scatter(
            x[is_highlight],
            y[is_highlight],
            c=colors[highlight],
            s=style.POINT_SIZE_BACKGROUND,
            linewidths=0,
        )
    for values, line in ((x, ax.axvline), (y, ax.axhline)):
        line(
            np.percentile(values, STRONG_CELL_PERCENTILE),
            color=style.COLORS["reference"],
            linestyle=style.LINE_STYLE_REFERENCE,
            linewidth=style.LINE_WIDTH_THIN,
        )


def celltype_legend(ax: plt.Axes, dataset: str, **kwargs) -> None:
    """Legend of `dataset`'s cell type colors."""
    ax.legend(
        handles=[
            Patch(color=color, label=celltype)
            for celltype, color in style.CELL_TYPE_COLORS[dataset].items()
        ],
        **kwargs,
    )


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
