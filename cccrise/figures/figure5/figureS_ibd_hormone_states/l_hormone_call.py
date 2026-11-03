"""How EE cells are called L-hormone+, as Figure 2 calls lung epithelial
cells by markers (`figure2._common.plot_dotplot`): EE cells detecting any
L hormone (GCG, PYY, INSL5) against the other EE, EC and tuft cells, by
L hormones, other EE hormones, pan-EE, EC and tuft markers. The call marks
L-hormone expression, not a cell type: EE hormone repertoires shift as cells
mature and move up the crypt-villus axis (Beumer, Gehart & Clevers 2020)."""

from ... import common
from .. import _common as S

AXES = ("half",)
SECTIONS = {
    "L hormones": ["GCG", "PYY", "INSL5"],
    "Other EE": ["SST", "GIP", "GHRL", "MLN"],
    "Pan-EE": ["CHGA", "CHGB"],
    "EC": ["TPH1", "TAC1"],
    "Tuft": ["POU2F3", "DCLK1", "TRPM5"],
}
GROUPS = [*S.EE_GROUPS, "Tuft"]


def plot(axs):
    (ax,) = axs
    cells = S.programs("enteroendocrine_cells")
    genes = [g for section in SECTIONS.values() for g in section]
    mean, detected = S.group_stats(cells[genes], cells["Group"], GROUPS)
    common.source_data(
        "l_hormone_call",
        mean.T.add_suffix(" mean").join(detected.T.add_suffix(" detected")),
        "Mean expression and fraction detected per group: EE cells detecting GCG, "
        "PYY or INSL5 (L-hormone+ EE), other EE, EC and tuft cells.",
    )
    S.dotplot(ax, mean, detected, SECTIONS)


def makeFigure():
    return common.standalone_fixed(__name__)
