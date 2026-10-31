"""Annotated cell type proportions of each IBD biopsy by inflammation
status, at either annotation level, drawn as the ALAD cohort supplement
draws ALAD's (`figure2.figureS_alad_cohort.celltype_proportions`)."""

import pandas as pd

from ... import common, style
from ...figure2.figureS_alad_cohort.celltype_proportions import composition_q, draw
from .. import _common as S

#: Annotation level -> (`celltype_counts` column, how captions name it)
LEVELS = {"lineage": ("lineage", "lineage"), "fine": ("celltype", "fine cell type")}


def proportions(column: str) -> pd.DataFrame:
    """Fraction of each biopsy's (rows) cells in each `column` value
    (columns), with the biopsy's inflammation "Status"."""
    counts = S.celltype_counts()
    fractions = pd.crosstab(
        counts[S.SPEC.condition_column],
        counts[column],
        values=counts["n_cells"],
        aggfunc="sum",
        normalize="index",
    ).fillna(0)
    status = S.samples().set_index(S.SPEC.condition_column)[S.GROUP_COLUMN]
    return fractions.assign(Status=status.reindex(fractions.index).to_numpy())


def ordered_celltypes(column: str) -> list[str]:
    """`column`'s values by lineage (`style.CELL_TYPES` order), then name."""
    counts = S.celltype_counts()[list(dict.fromkeys(["lineage", column]))]
    counts = counts.drop_duplicates()
    rank = {name: i for i, name in enumerate(style.CELL_TYPES[S.DATASET])}
    pairs = sorted(
        zip(counts["lineage"], counts[column]), key=lambda p: (rank[p[0]], p[1])
    )
    return list(dict.fromkeys(name for _, name in pairs))


def plot_level(ax, level: str) -> None:
    column, noun = LEVELS[level]
    df = proportions(column)
    celltypes = ordered_celltypes(column)
    q = composition_q(df, celltypes, list(S.GROUP_LABELS))
    common.source_data(
        f"{level}_fraction_per_biopsy",
        df.rename_axis("Biopsy")
        .reset_index()
        .melt(
            id_vars=["Biopsy", "Status"], var_name="Cell type", value_name="Fraction"
        ),
        f"Fraction of each terminal ileum biopsy's cells annotated as each {noun}, "
        "with the biopsy's inflammation status.",
    )
    common.source_data(
        f"{level}_fraction_kruskal",
        pd.DataFrame({"Cell type": celltypes, "Kruskal-Wallis q": q}),
        f"Kruskal-Wallis test of each {noun}'s per-biopsy fraction across the "
        "three inflammation statuses; p-values Benjamini-Hochberg corrected "
        f"across {noun}s (q).",
    )
    counts = df["Status"].value_counts()
    labels = {g: f"{label} (n={counts[g]})" for g, label in S.GROUP_LABELS.items()}
    draw(ax, df, celltypes, q, S.GROUP_COLORS, labels, vertical=level == "fine")
