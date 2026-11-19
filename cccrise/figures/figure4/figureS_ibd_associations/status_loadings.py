"""Read-depth-corrected biopsy loadings of IBD CPD components 3 and 4 per
inflammation-status group, with one-way ANOVA statistics.

Read-depth-corrected loadings of the CPD components associated with
inflammation in CD (C3, C4) per group, as violins with each biopsy as a point and the median marked, with
each test's ANOVA statistics, as Figure 3A shows ALAD's."""

import numpy as np
import pandas as pd

from ... import common, style
from .. import _common as S
from ._common import regrouping

#: (association, CP component) pairs drawn
TESTS = [
    ("CD uninflamed vs. inflamed", 3),
    ("CD uninflamed vs. inflamed", 4),
]
AXES = ("square",) * len(TESTS)
#: y-axis room above the data for the ANOVA statistics
STATS_HEADROOM = 1.4
#: Each group's color: the status colors, and CD (both statuses) apart
GROUP_COLORS = S.GROUP_COLORS | {"CD": style.CATEGORICAL[3]}
ANOVA_COLUMNS = ["Association", "Component", "F", "P", "PAdj", "EtaSquared"]
POINT_ALPHA = 0.35
JITTER = 0.12


def plot(axs):
    table = S.association_table("cp_A")
    points = loading_points()
    for ax, (name, k) in zip(axs, TESTS, strict=True):
        rows = points[(points["Association"] == name) & (points["Component"] == k)]
        draw_violins(ax, rows, levels(association(name)))
        ax.set_ylim(top=STATS_HEADROOM * ax.get_ylim()[1])
        ax.set_ylabel(f"CPD component {k} loading")
        ax.text(
            0.03,
            0.97,
            common._anova_stats(table[table["Association"] == name], k),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=style.FONT_SIZE_DENSE,
        )
    _source_data(points, table)


def draw_violins(ax, rows: pd.DataFrame, groups: list[str]) -> None:
    """One violin per group of `rows`' loadings, the biopsies jittered over
    it and its median as a tick."""
    values = [rows.loc[rows["Group"] == g, "Loading"].to_numpy() for g in groups]
    x = np.arange(len(groups))
    bodies = ax.violinplot(values, positions=x, widths=0.7, showextrema=False)
    for body, group in zip(bodies["bodies"], groups, strict=True):
        body.set(facecolor=GROUP_COLORS[group], edgecolor="none", alpha=0.8)
    rng = np.random.default_rng(0)
    for i, v in enumerate(values):
        ax.scatter(
            i + rng.uniform(-JITTER, JITTER, len(v)),
            v,
            color=style.COLORS["ink"],
            s=style.POINT_SIZE_CELL,
            alpha=POINT_ALPHA,
            linewidths=0,
            zorder=3,
        )
    ax.scatter(
        x,
        [np.median(v) for v in values],
        marker="_",
        s=4 * style.POINT_SIZE,
        color=style.COLORS["ink"],
        linewidths=style.LINE_WIDTH,
        zorder=4,
    )
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_xticks(
        x,
        [common.level_label(g).replace(" ", "\n") for g in groups],
        fontsize=style.FONT_SIZE_DENSE,
    )


def association(name: str) -> dict:
    return next(spec for spec in S.ASSOCIATIONS if spec["name"] == name)


def levels(spec: dict) -> list[str]:
    """The association's groups, in drawing order."""
    if "levels" in spec:
        return spec["levels"]
    if "other_label" in spec:
        return [spec["control_label"], spec["other_label"]]
    return list(S.GROUP_LABELS)


def loading_points() -> pd.DataFrame:
    """Each drawn test's biopsies: loading and group."""
    adata = S.condition_adata(S.corrected_loadings("cp_A"))
    points = pd.concat(
        common.group_loadings(
            adata,
            k,
            S.GROUP_COLUMN,
            S.SPEC.condition_column,
            read_depth_correct=False,
            **regrouping(association(name)),
        ).assign(Association=name, Component=k)
        for name, k in TESTS
    )
    return points.rename(columns={"Condition": "Biopsy", "Status": "Group"})


def _source_data(points: pd.DataFrame, table: pd.DataFrame) -> None:
    common.source_data(
        "cpd_loadings_by_status",
        points,
        "Each biopsy's read-depth-corrected loading on each drawn CPD component "
        "and its group in the association drawn with it.",
    )
    tested = pd.concat(
        table[(table["Association"] == name) & (table["Component"] == k)]
        for name, k in TESTS
    )
    common.source_data(
        "cpd_status_anova",
        tested[ANOVA_COLUMNS],
        "One-way ANOVA of each drawn component's loadings over the association's "
        "groups: F, p, p Benjamini-Hochberg adjusted across every component and "
        "association tested (PAdj), and eta-squared.",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
