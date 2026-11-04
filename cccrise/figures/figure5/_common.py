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

from ...anova import run_multiple_factors_anova
from ...factors import load_factors_netcdf
from ...runners.datasets import DATASETS, IBD_PROGRAMS_DIR, VALIDATION_DIR
from ...utils import correct_by_read_depth
from .. import common
from ..figure2._common import DOT_FRACTIONS, DOT_SIZE_MAX, label_sections

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
#: An EE cell is in a hormone state when its sender weight on that program is
#: in the top `STATE_QUARTILE` of EE cells and not in the other's
STATE_QUARTILE = 0.25
STATE_NAMES = {3: "C3-high (villus-like)", 4: "C4-high (crypt-like)"}
#: Every EE cell by which programs it is in the top `STATE_QUARTILE` of
CLASS_NAMES = ["C3 only", "C3 + C4", "C4 only", "Neither"]
ZOOM_NOTE = "zoom: EE, EC, tuft"
DODGE = 0.27  # x offset between the status groups' boxes
JITTER = 0.08


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
def celltype_weights() -> pd.DataFrame:
    """Mean per-cell sender and receiver weight of each lineage and fine cell
    type on every CP component (`run_cohort.celltype_weights`)."""
    return _load("celltype_component_weights")


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


def hormone_classes(cells: pd.DataFrame) -> pd.Series:
    """Each EE cell's `CLASS_NAMES` class: whether its C3 and C4 sender
    weights are in the top `STATE_QUARTILE` of EE cells'; non-EE cells are
    NaN."""
    ee = cells["Cell type"] == "Enteroendocrine cells"
    high = {
        k: cells[f"C{k} sender weight"]
        >= cells.loc[ee, f"C{k} sender weight"].quantile(1 - STATE_QUARTILE)
        for k in STATE_NAMES
    }
    key = high[3].map({True: "3", False: ""}) + high[4].map({True: "4", False: ""})
    return key.map(dict(zip(["3", "34", "4", ""], CLASS_NAMES, strict=True))).where(ee)


def hormone_states(cells: pd.DataFrame) -> pd.Series:
    """Each EE cell's hormone state (`STATE_NAMES`: high on one program and
    not the other, else "Other EE"; `hormone_classes`); non-EE cells NaN."""
    exclusive = {CLASS_NAMES[0]: STATE_NAMES[3], CLASS_NAMES[2]: STATE_NAMES[4]}
    classes = hormone_classes(cells)
    return classes.map(exclusive).fillna("Other EE").where(classes.notna())


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


def status_legend(ax, side: str = "right", **kwargs) -> None:
    common.legend(
        ax,
        side,
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
        ax.set_title(title, fontsize=common.style.FONT_SIZE_LABEL)


def weight_map(ax, xy: np.ndarray, values: np.ndarray, quantile: float = 0.99):
    """`values` on a PaCMAP, diverging and symmetric at their |`quantile`|,
    highest drawn on top; returns the points for a colorbar."""
    order = np.argsort(values)
    limit = np.quantile(np.abs(values), quantile)
    return ax.scatter(
        *xy[order].T,
        c=values[order],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        s=common.style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
