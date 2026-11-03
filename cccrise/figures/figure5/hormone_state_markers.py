"""The two hormone states the programs resolve inside the EE annotation, as
Figure 2's marker dot plot: EE cells in the top quarter of C3 or C4 sender
weight (`_common.hormone_states`) by BMP targets (the villus-tip signal) and
hormones. Genes shown differ between the states (Mann-Whitney, BH q <
`common.SIG_ALPHA`). C3-high cells are BMP-high and PYY/NTS/SCT-high,
villus-like; C4-high cells GCG-high, crypt-like (Beumer et al. 2018)."""

import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

from .. import common
from . import _common as S

AXES = ("mid",)
SECTIONS = {
    "BMP targets": ["ID1", "ID2", "ID3", "SMAD6", "SMAD7", "BAMBI"],
    "Hormones": ["GCG", "PYY", "NTS", "SCT", "CCK"],
}


def state_genes(cells: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
    """Each `SECTIONS` gene's Mann-Whitney p and BH q, C3-high vs C4-high."""
    genes = [g for section in SECTIONS.values() for g in section]
    a, b = (cells.loc[state == S.STATE_NAMES[k], genes] for k in (3, 4))
    p = [mannwhitneyu(a[g], b[g]).pvalue for g in genes]
    return pd.DataFrame({"Gene": genes, "p": p, "q": false_discovery_control(p)})


def plot(axs):
    (ax,) = axs
    cells = S.programs("enteroendocrine_cells")
    state = S.hormone_states(cells)
    tests = state_genes(cells, state).set_index("Gene")
    shown = tests.index[tests["q"] < common.SIG_ALPHA]
    order = [S.STATE_NAMES[3], S.STATE_NAMES[4]]
    mean, detected = S.group_stats(
        cells.loc[state.isin(order), shown], state[state.isin(order)], order
    )
    common.source_data(
        "hormone_state_markers",
        mean.T.add_suffix(" mean").join(detected.T.add_suffix(" detected")).join(tests),
        f"Mean expression and fraction detected of BMP targets and hormones in "
        f"EE cells in the top {S.STATE_QUARTILE:.0%} of C3 (not C4) or C4 (not "
        "C3) sender weight, with Mann-Whitney p and BH q between the two; genes "
        f"with q < {common.SIG_ALPHA} are drawn.",
    )
    S.dotplot(
        ax,
        mean,
        detected,
        {n: [g for g in genes if g in shown] for n, genes in SECTIONS.items()},
    )


def makeFigure():
    return common.standalone_fixed(__name__)
