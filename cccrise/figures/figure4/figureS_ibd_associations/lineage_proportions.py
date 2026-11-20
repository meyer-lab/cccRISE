"""Each IBD biopsy's lineage proportions by inflammation status, with each
lineage's Kruskal-Wallis q across the three statuses (Benjamini-Hochberg
corrected across lineages), drawn as the ALAD cohort supplement draws ALAD's
(`figure3.figureS_alad_associations.celltype_proportions`)."""

import pandas as pd

from ... import common, style
from ...figure3.figureS_alad_associations.celltype_proportions import (
    composition_q,
    draw,
)
from .. import _common as S

AXES = ("double_rect",)  # widened by `match_row_extent`


def plot(axs):
    df = proportions()
    lineages = ordered_lineages()
    q = composition_q(df, lineages, list(S.GROUP_LABELS))
    common.source_data(
        "lineage_fraction_per_biopsy",
        df.rename_axis("Biopsy")
        .reset_index()
        .melt(
            id_vars=["Biopsy", "Status"], var_name="Cell type", value_name="Fraction"
        ),
        "Fraction of each terminal ileum biopsy's cells annotated as each lineage, "
        "with the biopsy's inflammation status.",
    )
    common.source_data(
        "lineage_fraction_kruskal",
        pd.DataFrame({"Cell type": lineages, "Kruskal-Wallis q": q}),
        "Kruskal-Wallis test of each lineage's per-biopsy fraction across the "
        "three inflammation statuses; p-values Benjamini-Hochberg corrected "
        "across lineages (q).",
    )
    counts = df["Status"].value_counts()
    labels = {g: f"{label} (n={counts[g]})" for g, label in S.GROUP_LABELS.items()}
    draw(axs[0], df, lineages, q, S.GROUP_COLORS, labels)


def proportions() -> pd.DataFrame:
    """Fraction of each biopsy's (rows) cells in each lineage (columns), with
    the biopsy's inflammation "Status"."""
    counts = S.celltype_counts()
    fractions = pd.crosstab(
        counts[S.SPEC.condition_column],
        counts["lineage"],
        values=counts["n_cells"],
        aggfunc="sum",
        normalize="index",
    ).fillna(0)
    status = S.samples().set_index(S.SPEC.condition_column)[S.GROUP_COLUMN]
    return fractions.assign(Status=status.reindex(fractions.index).to_numpy())


def ordered_lineages() -> list[str]:
    """The cohort's lineages, in `style.CELL_TYPES` order."""
    present = set(S.celltype_counts()["lineage"])
    return [name for name in style.CELL_TYPES[S.DATASET] if name in present]


def makeFigure():
    return common.standalone_fixed(__name__)
