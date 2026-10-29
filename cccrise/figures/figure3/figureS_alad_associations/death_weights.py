"""Condition loadings of the CP components significantly associated with
death, per group, as Fig. 3 shows the 6-month outcome association."""

import pandas as pd

from ... import common
from .. import _common as S
from ..outcome_association import outcome_points, plot_outcome_weights
from ._common import TABLE_COLUMNS

#: (association, CP component) pairs significant after FDR correction
TESTS = (
    ("6-Month Died vs. Alive", 4),
    ("1-Year Died vs. Alive", 4),
)
AXES = ("square",) * len(TESTS)


def plot(axs):
    for i, (ax, (association, component)) in enumerate(zip(axs, TESTS, strict=True)):
        plot_outcome_weights(ax, association, component)
        save_weights(i, association, component)
    table = S.association_table()
    tested = pd.concat(
        [
            table[(table["Association"] == a) & (table["Component"] == c)]
            for a, c in TESTS
        ]
    )
    common.source_data(
        "anova_component_death",
        tested[[c for c in TABLE_COLUMNS if c in tested]],
        "The one-way ANOVA shown on each panel: one row per (association, CP "
        "component). P is the raw p-value; PAdj is Benjamini-Hochberg "
        "FDR-corrected across all associations and components.",
    )


def save_weights(part: int, association: str, component: int) -> None:
    """The per-sample loadings `plot_outcome_weights` draws on axes `part`."""
    name = association.split(" ")[0].lower().replace("-", "_")
    common.source_data(
        f"loading_component{component}_{name}_death",
        outcome_points(association, component).rename(
            columns={
                "Condition": "Sample",
                "Status": "Group",
                "Loading": f"Component {component} loading",
            }
        ),
        f"One row per BAL sample in '{association}': its group and CP component "
        f"{component}'s read-depth-corrected condition loading (the points and "
        "box plot; a patient's samples are joined by a line).",
        part=part,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
