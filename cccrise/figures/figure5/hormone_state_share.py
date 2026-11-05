"""Per biopsy (at least `MIN_EE` EE cells), the share of its EE cells in the
top quarter of C3 sender weight only (villus-like), of both C3 and C4, of C4
only (crypt-like) or of neither (`_common.hormone_classes`), by inflammation
status (Kruskal-Wallis p): inflammation removes C3-only cells, not C3-high
ones."""

import pandas as pd

from .. import common
from . import _common as S

AXES = ("rect",)
MIN_EE = 3
LABELS = {
    "C3 only": "C3 only\n(villus-like)",
    "C3 + C4": "C3 + C4",
    "C4 only": "C4 only\n(crypt-like)",
    "Neither": "Neither",
}


def class_shares(classify=S.hormone_classes) -> pd.DataFrame:
    """Per biopsy with at least `MIN_EE` classified cells, the share of them
    in each class `classify` gives (NaN: not classified)."""
    cells = S.programs("enteroendocrine_cells")
    state = classify(cells)
    ee = state.notna()
    shares = pd.crosstab(cells.loc[ee, "Biopsy"], state[ee], normalize="index")
    counts = cells.loc[ee, "Biopsy"].value_counts()
    shares = shares[counts.reindex(shares.index) >= MIN_EE]
    status = S.samples().set_index(S.SPEC.condition_column)[S.GROUP_COLUMN]
    return shares.assign(Status=status.reindex(shares.index).to_numpy())


def plot(axs):
    (ax,) = axs
    shares = class_shares()
    common.source_data(
        "hormone_class_share_by_biopsy",
        shares.rename_axis("Biopsy").reset_index(),
        f"Share of each biopsy's EE cells (biopsies with at least {MIN_EE}) in "
        f"the top {S.STATE_QUARTILE:.0%} of C3 sender weight only, of both C3 and "
        "C4, of C4 only, or of neither, with the biopsy's inflammation status; "
        "Kruskal-Wallis p across statuses above each class.",
    )
    S.status_boxes(
        ax,
        {
            LABELS[name]: [
                shares.loc[shares["Status"] == g, name] for g in S.GROUP_LABELS
            ]
            for name in S.CLASS_NAMES
        },
    )
    ax.set_ylabel("Share of biopsy's\nEE cells")
    # Top right, clear of the Neither boxes' whiskers
    S.status_legend(
        ax,
        loc="upper right",
        bbox_to_anchor=(0.74, 1),
        labelspacing=0.2,
        borderaxespad=0,
        borderpad=0.2,
        handlelength=1,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
