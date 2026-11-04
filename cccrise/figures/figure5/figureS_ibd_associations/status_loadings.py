"""Read-depth-corrected loadings of the status-associated CP components per
group, with each test's ANOVA statistics, as Figure 3A shows ALAD's."""

import pandas as pd

from ... import common, style
from .. import _common as S
from ._common import regrouping

#: (association, CP component) pairs drawn
TESTS = [
    ("Inflammation status (3-way)", 1),
    ("Healthy vs. CD", 1),
    ("CD uninflamed vs. inflamed", 3),
    ("CD uninflamed vs. inflamed", 4),
]
AXES = ("square",) * len(TESTS)
#: y-axis room above the data for the ANOVA statistics
STATS_HEADROOM = 1.4
#: Each group's box color: the status colors, and CD (both statuses) apart
GROUP_COLORS = S.GROUP_COLORS | {"CD": style.CATEGORICAL[3]}
ANOVA_COLUMNS = ["Association", "Component", "F", "P", "PAdj", "EtaSquared"]


def plot(axs):
    table = S.association_table("cp_A")
    adata = S.condition_adata(S.corrected_loadings("cp_A"))
    for ax, (name, k) in zip(axs, TESTS, strict=True):
        spec = association(name)
        common.plot_category_component_weights(
            adata,
            k,
            S.GROUP_COLUMN,
            ax,
            xlabel="",
            anova_df=table[table["Association"] == name],
            condition_column=S.SPEC.condition_column,
            palette=[GROUP_COLORS[g] for g in levels(spec)],
            read_depth_correct=False,
            show_counts=False,
            **regrouping(spec),
        )
        ax.set_ylabel(f"CP component {k} loading")
        ax.set_xticks(
            ax.get_xticks(),
            [t.get_text().replace(" ", "\n") for t in ax.get_xticklabels()],
            fontsize=style.FONT_SIZE_DENSE,
        )
        ax.set_ylim(top=STATS_HEADROOM * ax.get_ylim()[1])
    _source_data(adata, table)


def association(name: str) -> dict:
    return next(spec for spec in S.ASSOCIATIONS if spec["name"] == name)


def levels(spec: dict) -> list[str]:
    """The association's groups, in drawing order."""
    if "levels" in spec:
        return spec["levels"]
    if "other_label" in spec:
        return [spec["control_label"], spec["other_label"]]
    return list(S.GROUP_LABELS)


def _source_data(adata, table: pd.DataFrame) -> None:
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
    common.source_data(
        "cp_loadings_by_status",
        points.rename(columns={"Condition": "Biopsy", "Status": "Group"}),
        "Each biopsy's read-depth-corrected loading on each drawn CP component "
        "and its group in the association drawn with it.",
    )
    tested = pd.concat(
        table[(table["Association"] == name) & (table["Component"] == k)]
        for name, k in TESTS
    )
    common.source_data(
        "cp_status_anova",
        tested[ANOVA_COLUMNS],
        "One-way ANOVA of each drawn component's loadings over the association's "
        "groups: F, p, p Benjamini-Hochberg adjusted across every component and "
        "association tested (PAdj), and eta-squared.",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
