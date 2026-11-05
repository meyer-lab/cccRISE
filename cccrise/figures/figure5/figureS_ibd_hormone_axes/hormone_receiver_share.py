"""`extreme_share` for the hormone programs' receivers. C3's receivers are
EC cells and C4's tuft cells, so the two sides never share a cell and each
type is split alone: per biopsy (at least `_common.MIN_EE` cells of the
type), the share of its EC cells in
their top `_common.STATE_QUARTILE` of C3 receiver weight, and of its tuft
cells in theirs of C4, by inflammation status (Kruskal-Wallis p)."""

from functools import partial

import pandas as pd

from ... import common
from .. import _common as S

AXES = ("rect_half",)  # as wide per slot as `extreme_share`
HIGH = "High"
#: component -> (receiver type, label)
RECEIVERS = {
    3: ("Enterochromaffin cells", "C3-high\nof EC cells"),
    4: ("Tuft cells", "C4-high\nof tuft cells"),
}


def receiver_classes(cells: pd.DataFrame, k: int) -> pd.Series:
    """`HIGH` for component `k`'s receiver cells in their top quarter of its
    receiver weight, else "Other"; other cells NaN."""
    carriers = cells["Cell type"] == RECEIVERS[k][0]
    weight = cells.loc[carriers, f"C{k} receiver weight"]
    high = weight >= weight.quantile(1 - S.STATE_QUARTILE)
    return high.map({True: HIGH, False: "Other"}).reindex(cells.index)


def plot(axs):
    (ax,) = axs
    shares = {k: S.class_shares(partial(receiver_classes, k=k)) for k in RECEIVERS}
    common.source_data(
        "hormone_receiver_share_by_biopsy",
        pd.concat(
            {f"C{k}": s[[HIGH, "Status"]] for k, s in shares.items()},
            names=["Component", "Biopsy"],
        ).reset_index(),
        f"Share of each biopsy's EC (C3) or tuft (C4) cells (biopsies with at least "
        f"{S.MIN_EE}) in that type's top {S.STATE_QUARTILE:.0%} of the component's "
        "receiver weight, with the biopsy's inflammation status; Kruskal-Wallis p "
        "across statuses above each.",
    )
    S.status_boxes(
        ax,
        {
            RECEIVERS[k][1]: [s.loc[s["Status"] == g, HIGH] for g in S.GROUP_LABELS]
            for k, s in shares.items()
        },
    )
    ax.set_ylabel("Share of biopsy's\nreceiver-type cells")


def makeFigure():
    return common.standalone_fixed(__name__)
