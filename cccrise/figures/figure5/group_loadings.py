"""Every IBD CP component's read-depth-corrected condition loading per
inflammation status (box: quartiles, whiskers to 1.5 IQR; points: biopsies),
as Figure 3 shows ALAD's outcome-associated component
(`figure3.outcome_association`). Above each component: the three-group
one-way ANOVA's FDR (`S.THREE_WAY`), and a bracket per pairwise contrast
with FDR < `common.SIG_ALPHA`; FDR is Benjamini-Hochberg across every
association and component tested (`S.association_table`). The axis is
symmetric-log, linear below `LINEAR_BELOW`, as loadings span five orders of
magnitude (one biopsy's C5 loading is slightly negative after correction)."""

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from .. import common, style
from ..figure3.figureS_alad_associations._common import TABLE_COLUMNS
from . import _common as S

LINEAR_BELOW = 1e-3
DODGE = 0.27  # x offset between the groups' boxes
BRACKET_STEP = 2.6  # factor between stacked brackets' heights
JITTER = 0.08
AXES = ("full",)


def loadings_by_group() -> pd.DataFrame:
    """One row per biopsy: its inflammation status and corrected loading on
    each CP component (`Component <k>`)."""
    loadings = S.corrected_loadings("cp_A").add_prefix("Component ")
    samples = S.samples()
    return pd.concat(
        [
            samples[[S.SPEC.condition_column, S.GROUP_COLUMN]].rename(
                columns={S.SPEC.condition_column: "Biopsy", S.GROUP_COLUMN: "Group"}
            ),
            loadings,
        ],
        axis=1,
    )


def plot(axs):
    (ax,) = axs
    table = S.association_table()
    points = loadings_by_group()
    _source_data(points, table)
    groups = list(S.GROUP_LABELS)
    components = range(1, len(points.filter(like="Component").columns) + 1)
    rng = np.random.default_rng(0)
    for k in components:
        values = [points.loc[points["Group"] == g, f"Component {k}"] for g in groups]
        draw_boxes(ax, k - 1, values, rng)
        _annotate(ax, k, max(v.max() for v in values), table)
    ax.set_yscale("symlog", linthresh=LINEAR_BELOW)
    lowest = points.filter(like="Component").to_numpy().min()
    ax.set_ylim(min(1.5 * lowest, -LINEAR_BELOW / 2), ax.get_ylim()[1] * BRACKET_STEP)
    ax.set_xticks(range(len(components)), components)
    ax.set_xlim(-0.6, len(components) - 0.4)
    ax.set_xlabel("CP component")
    ax.set_ylabel("Corrected loading")
    counts = points["Group"].value_counts()
    common.legend(
        ax,
        "above",
        handles=[
            Patch(color=S.GROUP_COLORS[g], alpha=0.8, label=f"{label} (n={counts[g]})")
            for g, label in S.GROUP_LABELS.items()
        ],
        ncols=len(S.GROUP_LABELS),
    )


def draw_boxes(ax, x: float, values: list[pd.Series], rng) -> None:
    """Boxes and jittered biopsies centered on `x`, one per inflammation
    status (`values`, in `S.GROUP_LABELS` order)."""
    positions = x + DODGE * (np.arange(len(values)) - (len(values) - 1) / 2)
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.8 * DODGE,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": style.COLORS["ink"], "lw": style.LINE_WIDTH},
        whiskerprops={"lw": style.LINE_WIDTH_THIN},
        capprops={"lw": style.LINE_WIDTH_THIN},
        boxprops={"lw": style.LINE_WIDTH_THIN},
    )
    for box, group in zip(boxes["boxes"], S.GROUP_LABELS, strict=True):
        box.set(facecolor=S.GROUP_COLORS[group], alpha=0.8)
    for position, group_values in zip(positions, values, strict=True):
        ax.scatter(
            position + rng.uniform(-JITTER, JITTER, len(group_values)),
            group_values,
            color=style.COLORS["ink"],
            s=style.POINT_SIZE_CELL,
            linewidths=0,
            zorder=3,
            rasterized=True,
        )


def _annotate(ax, component: int, top: float, table: pd.DataFrame) -> None:
    """The three-group FDR above `component`, and a bracket per significant
    pairwise contrast, stacked from `top` up."""
    rows = table[table["Component"] == component].set_index("Association")
    x = component - 1
    offset = {g: DODGE * (i - 1) for i, g in enumerate(S.GROUP_LABELS)}
    height = top
    for name, (low, high) in S.CONTRASTS.items():
        if not rows.loc[name, "PAdj"] < common.SIG_ALPHA:
            continue
        height *= BRACKET_STEP
        x0, x1 = x + offset[low], x + offset[high]
        ax.plot(
            [x0, x0, x1, x1],
            [height / 1.3, height, height, height / 1.3],
            color=style.COLORS["ink"],
            lw=style.LINE_WIDTH_THIN,
        )
        ax.text((x0 + x1) / 2, height, "*", ha="center", va="center", fontweight="bold")
    q = rows.loc[S.THREE_WAY, "PAdj"]
    significant = q < common.SIG_ALPHA
    ax.text(
        x,
        1,
        f"q = {q:.1g}",
        transform=blended_transform_factory(ax.transData, ax.transAxes),
        ha="center",
        va="bottom",
        fontsize=style.FONT_SIZE_DENSE,
        fontweight="bold" if significant else "normal",
    )


def _source_data(points: pd.DataFrame, table: pd.DataFrame) -> None:
    common.source_data(
        "cp_loadings_by_inflammation",
        points,
        "Each terminal ileum biopsy's inflammation status and read-depth-"
        "corrected loading on every CP component (the points and boxes).",
    )
    common.source_data(
        "cp_inflammation_anova",
        table[[c for c in TABLE_COLUMNS if c in table]],
        "One-way ANOVA of each CP component's corrected loadings across "
        "inflammation status and in each pairwise contrast: F, p, p "
        "Benjamini-Hochberg adjusted across every component and association "
        "tested (PAdj), and eta-squared.",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
