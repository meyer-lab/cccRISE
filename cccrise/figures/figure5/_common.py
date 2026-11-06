"""The dataset Figure 5 and its supplements draw -- the IBDverse terminal
ileum cohort (the registry's `ibd` entry: RISE 60 / CP 6, mesenchymal lineage
excluded, at most 2,000 cells per biopsy) -- the `cccrise-cohort` tables that
summarize its biopsies and cell types, the `cccrise-ibd-programs` tables
behind the program panels, the inflammation-status associations tested
against its components (as Figure 3 tests ALAD's), and the drawing helpers
those panels share."""

from functools import cache
from typing import Any

import anndata
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from scipy.stats import kruskal

from ...anova import adjust_pvalues, run_multiple_factors_anova
from ...factors import load_factors_netcdf
from ...runners.datasets import DATASETS, IBD_PROGRAMS_DIR, VALIDATION_DIR
from ...utils import correct_by_read_depth
from .. import common
from ..figure2._common import DOT_FRACTIONS, DOT_SIZE_MAX, GROUP_TAU, label_sections

DATASET = "ibd"
SPEC = DATASETS[DATASET]
GROUP_COLUMN = "inflammation_status"

#: `inflammation_status` levels, in disease order, and how figures name them
GROUP_LABELS = {
    "Healthy": "Healthy",
    "CD_uninflamed": "CD uninflamed",
    "CD_inflamed": "CD inflamed",
}
GROUP_COLORS = dict(zip(GROUP_LABELS, common.style.CATEGORICAL, strict=False))
_UNINFLAMED, _INFLAMED = list(GROUP_LABELS)[1:]

#: Inflammation-status associations tested against every component, as
#: `figure3._common.ASSOCIATIONS` (see `anova.prepare_group_labels`)
ASSOCIATIONS: list[dict[str, Any]] = [
    {"name": "Inflammation status (3-way)", "group_column": GROUP_COLUMN},
    {
        "name": "Healthy vs. CD",
        "group_column": GROUP_COLUMN,
        "control_label": "Healthy",
        "other_label": "CD",
    },
    {
        "name": "CD uninflamed vs. inflamed",
        "group_column": GROUP_COLUMN,
        "levels": [_UNINFLAMED, _INFLAMED],
        "control_label": _UNINFLAMED,
    },
]

#: White outline keeping map labels legible over their clusters
LABEL_HALO = [withStroke(linewidth=1.5, foreground="white")]
#: The annotated types on the zoomed and all-cell maps
MAP_GROUPS = {
    "EE": common.style.CATEGORICAL[3],
    "EC": common.style.CATEGORICAL[0],
    "Tuft": common.style.CATEGORICAL[2],
}
#: The inflammatory programs' colors, off the status and baseline colors
PROGRAM_COLORS = {1: common.style.CATEGORICAL[4], 5: common.style.CATEGORICAL[5]}
#: The hormone programs' colors in line plots
HORMONE_COLORS = {3: common.style.COLORS["up"], 4: common.style.COLORS["emphasis"]}
#: Per-biopsy shares are over biopsies with at least this many classified cells
MIN_EE = 3
#: Each EE cell's C3 and C4 sender weights over their largest EE value give a
#: magnitude and, where they sum above 0, an angle (0 degrees C3 only, 90 C4
#: only, -45 to 135; `hormone_angles`). Cells past `EXTREME_FLOOR` (Figure 2's
#: cutoff) at angles up to `EXTREME_SECTOR` are fully in C3's state, from
#: 90 - `EXTREME_SECTOR` in C4's, others past it mixed; the rest (below it,
#: or summing to at most 0) other
EXTREME_FLOOR = GROUP_TAU
EXTREME_SECTOR = 30
EXTREME_NAMES = ["Villus-like", "Mixed", "Crypt-like", "Other"]
EXTREME_LABELS = {
    "Villus-like": "Villus-like (Cmp. 3)",
    "Mixed": "Mixed",
    "Crypt-like": "Crypt-like (Cmp. 4)",
    "Other": "Other",
}
#: The angle ranges the extreme classes take, as the angle histogram colors
#: and keys them
SECTOR_COLORS = {
    "Cmp. 3 high": HORMONE_COLORS[3],
    "Both": common.style.COLORS["muted"],
    "Cmp. 4 high": HORMONE_COLORS[4],
}
ZOOM_NOTE = "zoom: EE, EC, tuft"
#: Expression maps color each cell by its expression over this quantile
EXPRESSION_QUANTILE = 0.99
EXPRESSION_LABEL = f"Expr. / {EXPRESSION_QUANTILE:.0%} quantile"
DODGE = 0.27  # x offset between the status groups' boxes
JITTER = 0.08
SHARE_PAD = 0.04  # y room past the 0-1 shares' range in `status_boxes`


def _load(table: str, runner: str = "cohort") -> pd.DataFrame:
    return common.load_runner_output(
        f"{table}_{SPEC.output_key}.parquet",
        VALIDATION_DIR if runner == "cohort" else IBD_PROGRAMS_DIR,
        hint=f"    uv run cccrise-{runner} --dataset {DATASET}",
    )


@cache
def programs(table: str) -> pd.DataFrame:
    """A `cccrise-ibd-programs` table (`run_ibd_programs`)."""
    return _load(table, "ibd-programs")


def map_group(group: pd.Series) -> pd.Series:
    """`run_ibd_programs.ee_group`'s groups as `MAP_GROUPS`: its split of EE
    cells by L-hormone detection merged back into "EE"."""
    return group.replace({"L-hormone+ EE": "EE", "Other EE": "EE"})


@cache
def samples() -> pd.DataFrame:
    """One row per biopsy, in the fit's condition order: its sample-level
    metadata, cell count, read depth, and raw and read-depth-corrected CP
    loadings (`run_cohort.sample_table`)."""
    return _load("cohort_samples")


@cache
def celltype_counts() -> pd.DataFrame:
    """Cells per biopsy x lineage x fine cell type."""
    return _load("cohort_celltype_counts")


@cache
def factors() -> xr.Dataset:
    return load_factors_netcdf(SPEC.factors_path())


def corrected_loadings(factor: str) -> pd.DataFrame:
    """Condition factor `factor` (`cp_A` or `pf2_A`), read-depth corrected
    as the ANOVA tests it (`utils.correct_by_read_depth`), one row per
    biopsy in `samples` order and one column per component (1-indexed)."""
    A = factors()[factor]
    assert np.array_equal(A["condition"].astype(str), samples()[SPEC.condition_column])
    values = correct_by_read_depth(A.values.astype(float), samples()["read_depth"])
    return pd.DataFrame(values, columns=range(1, values.shape[1] + 1))


def condition_adata(values: pd.DataFrame | np.ndarray) -> anndata.AnnData:
    """One observation per biopsy (`samples` order) carrying `values` as its
    condition factor (`uns["A"]`): what `anova` and
    `common.plot_category_component_weights` need, without the cells."""
    obs = samples().drop(columns=samples().filter(like="loading").columns)
    obs.index = obs[SPEC.condition_column].astype(str).to_numpy()
    return anndata.AnnData(obs=obs, uns={"A": np.asarray(values, dtype=float)})


@cache
def association_table(factor: str = "cp_A") -> pd.DataFrame:
    """One-way ANOVA of every component's read-depth-corrected `factor`
    loading (`cp_A` or `pf2_A`) against every association, FDR-corrected
    across all associations and components, as `figure3._common` tests ALAD's."""
    return run_multiple_factors_anova(
        condition_adata(corrected_loadings(factor)),
        ASSOCIATIONS,
        condition_column=SPEC.condition_column,
        correction="fdr_bh",
        global_correction=True,
        read_depth_correct=False,
    )


def hormone_angles(cells: pd.DataFrame) -> pd.DataFrame:
    """EE cells' C3 and C4 sender weights over their largest EE value, their
    magnitude, and their angle (degrees; NaN where the two sum to at most 0,
    whose angles wrap)."""
    ee = cells.loc[cells["Cell type"] == "Enteroendocrine cells"]
    relative = {
        k: ee[f"C{k} sender weight"] / ee[f"C{k} sender weight"].max()
        for k in HORMONE_COLORS
    }
    angle = np.degrees(np.arctan2(relative[4], relative[3]))
    return pd.DataFrame(
        {
            "C3 relative": relative[3],
            "C4 relative": relative[4],
            "Magnitude": np.hypot(relative[3], relative[4]),
            "Angle": angle.where(relative[3] + relative[4] > 0),
        }
    )


def hormone_extremes(cells: pd.DataFrame, floor: float = EXTREME_FLOOR) -> pd.Series:
    """Each EE cell's `EXTREME_NAMES` class from its `hormone_angles`, past
    magnitude `floor`; non-EE cells are NaN."""
    polar = hormone_angles(cells)
    strong = (polar["Magnitude"] >= floor) & polar["Angle"].notna()
    villus, mixed, crypt, low = EXTREME_NAMES
    name = np.select(
        [
            strong & (polar["Angle"] <= EXTREME_SECTOR),
            strong & (polar["Angle"] >= 90 - EXTREME_SECTOR),
            strong,
        ],
        [villus, crypt, mixed],
        low,
    )
    return pd.Series(name, index=polar.index).reindex(cells.index)


def class_shares(classify) -> pd.DataFrame:
    """Per biopsy with at least `MIN_EE` classified cells, the share of them
    in each class `classify(cells)` gives (NaN: not classified), with the
    biopsy's status."""
    cells = programs("enteroendocrine_cells")
    state = classify(cells)
    known = state.notna()
    shares = pd.crosstab(cells.loc[known, "Biopsy"], state[known], normalize="index")
    counts = cells.loc[known, "Biopsy"].value_counts()
    shares = shares[counts.reindex(shares.index) >= MIN_EE]
    status = samples().set_index(SPEC.condition_column)[GROUP_COLUMN]
    return shares.assign(Status=status.reindex(shares.index).to_numpy())


def cell_histogram(ax, values: pd.Series, bins, color=None) -> list:
    """Cells' `values` as one histogram; returns the bars, to recolor."""
    _, _, bars = ax.hist(
        values,
        bins=bins,
        color=color or common.style.COLORS["muted"],
        edgecolor=common.style.COLORS["ink"],
        linewidth=common.style.LINE_WIDTH_THIN,
    )
    ax.set_ylabel("EE cells")
    return list(bars)


def status_boxes(ax, slots: dict[str, list[pd.Series]]) -> None:
    """One `draw_boxes` slot per x label (`slots`' keys; values per status),
    its Kruskal-Wallis q across statuses (Benjamini-Hochberg corrected across
    slots) above it, bold if significant. Each slot is one x unit, so axes as
    wide per slot draw identical boxes."""
    rng = np.random.default_rng(0)
    q = adjust_pvalues([kruskal(*values).pvalue for values in slots.values()])
    for x, values in enumerate(slots.values()):
        draw_boxes(ax, x, values, rng)
        ax.text(
            x,
            1,
            f"q = {q[x]:.1g}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=common.style.FONT_SIZE_DENSE,
            fontweight="bold" if q[x] < common.SIG_ALPHA else "normal",
        )
    ax.set_xticks(range(len(slots)), list(slots), fontsize=common.style.FONT_SIZE_DENSE)
    ax.set_xlim(-0.5, len(slots) - 0.5)
    ax.set_ylim(-SHARE_PAD, 1 + SHARE_PAD)


def draw_boxes(ax, x: float, values: list[pd.Series], rng) -> None:
    """Boxes and jittered biopsies centered on `x`, one per inflammation
    status (`values`, in `GROUP_LABELS` order)."""
    positions = x + DODGE * (np.arange(len(values)) - (len(values) - 1) / 2)
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.8 * DODGE,
        showfliers=False,
        patch_artist=True,
        medianprops={
            "color": common.style.COLORS["ink"],
            "lw": common.style.LINE_WIDTH,
        },
        whiskerprops={"lw": common.style.LINE_WIDTH_THIN},
        capprops={"lw": common.style.LINE_WIDTH_THIN},
        boxprops={"lw": common.style.LINE_WIDTH_THIN},
    )
    for box, group in zip(boxes["boxes"], GROUP_LABELS, strict=True):
        box.set(facecolor=GROUP_COLORS[group], alpha=0.8)
    for position, group_values in zip(positions, values, strict=True):
        ax.scatter(
            position + rng.uniform(-JITTER, JITTER, len(group_values)),
            group_values,
            color=common.style.COLORS["ink"],
            s=common.style.POINT_SIZE_CELL,
            linewidths=0,
            zorder=3,
            rasterized=True,
        )


def status_legend(ax, **kwargs) -> None:
    """The inflammation statuses' key inside `ax`, placed by `kwargs`."""
    ax.legend(
        handles=[
            Patch(color=GROUP_COLORS[g], alpha=0.8, label=label)
            for g, label in GROUP_LABELS.items()
        ],
        fontsize=common.style.FONT_SIZE_DENSE,
        **kwargs,
    )


def group_stats(
    X: pd.DataFrame, group: pd.Series, order: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per group (rows, in `order`): each gene's mean and fraction detected."""
    keys = group.to_numpy()
    return X.groupby(keys).mean().reindex(order), (X > 0).groupby(keys).mean().reindex(
        order
    )


def dotplot(
    ax, mean: pd.DataFrame, detected: pd.DataFrame, sections: dict[str, list[str]]
):
    """Figure 2's marker dot plot (`figure2._common.plot_dotplot`) for any
    groups: dot size the fraction detecting a gene, color its mean over the
    gene's largest group mean, genes under `sections` names."""
    fontsize = common.style.FONT_SIZE_DENSE
    genes = label_sections(ax, sections, fontsize)
    relative = mean[genes] / mean[genes].max()
    x, y = np.meshgrid(np.arange(len(genes)), np.arange(len(mean)), indexing="ij")
    points = ax.scatter(
        x.ravel(),
        y.ravel(),
        s=DOT_SIZE_MAX * detected[genes].T.to_numpy().ravel(),
        c=relative.T.to_numpy().ravel(),
        cmap=common.style.CMAPS["weight"],
        vmin=0,
        vmax=1,
        linewidths=0,
    )
    ax.set_yticks(range(len(mean)), mean.index, fontsize=fontsize)
    ax.set_ylim(len(mean) - 0.5, -0.5)
    common.colorbar(ax, points, "Mean expr.\n/ max group")
    common.legend(
        ax,
        "right",
        handles=[
            ax.scatter([], [], s=DOT_SIZE_MAX * f, color=common.style.COLORS["ink"])
            for f in DOT_FRACTIONS
        ],
        labels=[f"{f:.0%}" for f in DOT_FRACTIONS],
        title="Detected",
    )


def map_axes(
    ax, title: str | None = None, xlabel: str = f"PaCMAP 1 ({ZOOM_NOTE})"
) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=common.style.FONT_SIZE_DENSE)
    ax.set_ylabel("PaCMAP 2")
    if title:
        # y fixed: a colorbar spanning stacked maps would push an auto title up
        ax.set_title(title, fontsize=common.style.FONT_SIZE_LABEL, y=1)


def weight_map(ax, xy: np.ndarray, values: np.ndarray, quantile: float = 0.99):
    """`values` on a PaCMAP, diverging and symmetric at their |`quantile`|,
    highest drawn on top; returns the points for a colorbar."""
    limit = np.quantile(np.abs(values), quantile)
    return _value_map(ax, xy, values, "RdBu_r", -limit, limit)


def relative_expression(values: np.ndarray) -> np.ndarray:
    """Expression `values` over their `EXPRESSION_QUANTILE`, capped at 1."""
    return np.clip(values / np.quantile(values, EXPRESSION_QUANTILE), 0, 1)


def expression_map(ax, xy: np.ndarray, relative: np.ndarray):
    """`relative_expression` on a PaCMAP, highest drawn on top; returns the
    points for a colorbar labeled `EXPRESSION_LABEL`."""
    return _value_map(ax, xy, relative, "Reds", 0, 1)


def _value_map(ax, xy, values, cmap, vmin, vmax):
    order = np.argsort(values)
    return ax.scatter(
        *xy[order].T,
        c=values[order],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=common.style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
