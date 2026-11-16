"""The EE cells fully in one hormone state (`_common.hormone_extremes`: C3's
villus-like and C4's crypt-like extremes, mixed between them, other below
the magnitude floor) as Figure 2's marker dot plot, by crypt and villus
markers, CCK (the I-cell step between them; Gehart et al. 2019) and BMP
targets (the villus-tip signal; Beumer et al. 2018). Genes shown differ
between the villus-like and crypt-like cells (Mann-Whitney, BH
q < `common.SIG_ALPHA`)."""

import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

from .. import common
from . import _common as S

AXES = ("mid_wide",)
SECTIONS = {
    "Crypt": ["GCG", "GPBAR1", "GPX2"],
    "Int.": ["CCK"],
    "Villus": ["PYY", "SCT", "NTS"],
    "BMP targets": ["ID1", "ID2", "ID3", "TBX3"],
}
EXTREMES = ("Villus-like", "Crypt-like")


def extreme_genes(cells: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
    """Each `SECTIONS` gene's Mann-Whitney p and BH q, villus-like vs
    crypt-like cells."""
    genes = [g for section in SECTIONS.values() for g in section]
    a, b = (cells.loc[state == group, genes].astype(float) for group in EXTREMES)
    p = [mannwhitneyu(a[g], b[g]).pvalue for g in genes]
    return pd.DataFrame({"Gene": genes, "p": p, "q": false_discovery_control(p)})


def plot(axs):
    (ax,) = axs
    cells = S.programs("enteroendocrine_cells")
    state = S.hormone_extremes(cells)
    tests = extreme_genes(cells, state).set_index("Gene")
    shown = tests.index[tests["q"] < common.SIG_ALPHA]
    mean, detected = S.group_stats(
        cells.loc[state.notna(), shown], state[state.notna()], S.EXTREME_NAMES
    )
    common.source_data(
        "hormone_extreme_markers",
        mean.T.add_suffix(" mean").join(detected.T.add_suffix(" detected")).join(tests),
        "Mean expression and fraction detected per EE class (villus-like: angle "
        f"at most {S.EXTREME_SECTOR} degrees, crypt-like: at least "
        f"{90 - S.EXTREME_SECTOR}, both magnitude at least {S.EXTREME_FLOOR}; "
        "mixed between; other below it), with Mann-Whitney p and BH q between "
        f"villus-like and crypt-like; genes with q < {common.SIG_ALPHA} are drawn.",
    )
    classed = state.notna()
    common.source_data(
        "hormone_extreme_marker_expression_per_cell",
        cells.loc[classed, ["Cell", "Biopsy", *tests.index]].assign(
            Class=state[classed]
        ),
        "Each classed EE cell's expression of every tested gene, the values behind "
        "the class means, fractions detected and Mann-Whitney tests.",
    )
    labels = [S.EXTREME_LABELS[name].replace(" (", "\n(") for name in S.EXTREME_NAMES]
    S.dotplot(
        ax,
        mean.set_axis(labels),
        detected.set_axis(labels),
        {n: [g for g in genes if g in shown] for n, genes in SECTIONS.items()},
        size_key="below",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
