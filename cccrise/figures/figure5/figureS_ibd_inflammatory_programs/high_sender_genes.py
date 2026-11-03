"""What the myeloid cells highest in C1 and C5 sender weight express, as
Figure 2's dot plot: the top `TOP_FRACTION` of myeloid cells by each
component's unscaled sender weight, split into C1-only, both and C5-only,
beside the other myeloid cells, by the genes they share (NF-kB / IL-1
activation) and the genes separating C1-only (IFN-gamma activation) from
C5-only cells (neutrophil-like genes)."""

import pandas as pd

from ... import common
from .. import _common as S

AXES = ("wide",)
TOP_FRACTION = 0.02
SECTIONS = {
    "Shared (NF-κB / IL-1)": [
        "IL1B",
        "SOD2",
        "NFKBIA",
        "TNFAIP3",
        "ICAM1",
        "CCL3",
        "CCL4",
        "CXCL8",
    ],
    "C1-only": ["CXCL10", "WARS", "KYNU", "IL4I1", "SERPINB9"],
    "C5-only": ["FCGR3B", "HCAR2", "CSF3R", "IL1R2", "S100A8", "G0S2"],
}
ORDER = ["C1 only", "C1 & C5", "C5 only", "Other"]


def groups(cells: pd.DataFrame) -> pd.Series:
    high = {
        k: cells[f"C{k} sender weight"]
        >= cells[f"C{k} sender weight"].quantile(1 - TOP_FRACTION)
        for k in (1, 5)
    }
    group = pd.Series("Other", index=cells.index)
    group[high[1] & high[5]] = "C1 & C5"
    group[high[1] & ~high[5]] = "C1 only"
    group[~high[1] & high[5]] = "C5 only"
    return group


def plot(axs):
    (ax,) = axs
    cells = S.programs("myeloid_program_cells")
    group = groups(cells)
    genes = [g for section in SECTIONS.values() for g in section]
    mean, detected = S.group_stats(cells[genes], group, ORDER)
    common.source_data(
        "high_sender_genes",
        mean.T.add_suffix(" mean").join(detected.T.add_suffix(" detected")),
        f"Mean expression and fraction detected in myeloid cells in the top "
        f"{TOP_FRACTION:.0%} of C1 and/or C5 sender weight, and in the other myeloid cells.",
    )
    S.dotplot(ax, mean, detected, SECTIONS)


def makeFigure():
    return common.standalone_fixed(__name__)
