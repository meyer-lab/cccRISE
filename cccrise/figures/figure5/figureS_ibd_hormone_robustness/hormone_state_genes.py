"""The two hormone states (`_common.hormone_states`) and their neighbors as
Figure 2's dot plot: the crypt-like GLP-1 state (GCG with the L-lineage
transcription factors ARX, ISL1, PAX6 and GPBAR1), the villus-like,
BMP-driven state (PYY, NTS, SCT, CCK), and other EE and EC markers. States
along the crypt-to-villus hormone switch (Beumer et al. 2018), not cell
types."""

from ... import common
from .. import _common as S

AXES = ("wide",)
SECTIONS = {
    "Crypt-like (GLP-1) state": ["GCG", "ARX", "ISL1", "PAX6", "GPBAR1"],
    "Villus-like (BMP) state": ["PYY", "INSL5", "NTS", "SCT", "CCK", "CHGB"],
    "Other EE / EC": ["SST", "GIP", "TPH1", "CHGA"],
}
ORDER = [S.STATE_NAMES[4], S.STATE_NAMES[3], "Other EE", "EC"]


def plot(axs):
    (ax,) = axs
    cells = S.programs("enteroendocrine_cells")
    group = S.hormone_states(cells).fillna(cells["Group"])
    genes = [g for section in SECTIONS.values() for g in section]
    keep = group.isin(ORDER)
    mean, detected = S.group_stats(cells.loc[keep, genes], group[keep], ORDER)
    common.source_data(
        "hormone_state_genes",
        mean.T.add_suffix(" mean").join(detected.T.add_suffix(" detected")),
        f"Mean expression and fraction detected per hormone state (EE cells in the "
        f"top {S.STATE_QUARTILE:.0%} of one program's sender weight and not the "
        "other's), the other EE cells and EC cells.",
    )
    S.dotplot(ax, mean, detected, SECTIONS)


def makeFigure():
    return common.standalone_fixed(__name__)
