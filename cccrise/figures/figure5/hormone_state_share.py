"""Per biopsy (at least `MIN_EE` EE cells), the share of its EE cells in the
top quarter of C3 sender weight only (villus-like), of both C3 and C4, of C4
only (crypt-like) or of neither (`_common.hormone_classes`), by inflammation
status (Kruskal-Wallis p): inflammation removes C3-only cells, not C3-high
ones."""

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from .. import common, style
from . import _common as S

AXES = ("rect",)
MIN_EE = 3
LABELS = {
    "C3 only": "C3 only\n(villus-like)",
    "C3 + C4": "C3 + C4",
    "C4 only": "C4 only\n(crypt-like)",
    "Neither": "Neither",
}


def class_shares() -> pd.DataFrame:
    cells = S.programs("enteroendocrine_cells")
    state = S.hormone_classes(cells)
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
    rng = np.random.default_rng(0)
    for x, name in enumerate(S.CLASS_NAMES):
        groups = [shares.loc[shares["Status"] == g, name] for g in S.GROUP_LABELS]
        S.draw_boxes(ax, x, groups, rng)
        p = kruskal(*groups).pvalue
        ax.text(
            x,
            1,
            f"p = {p:.1g}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold" if p < common.SIG_ALPHA else "normal",
        )
    ax.set_xticks(
        range(len(LABELS)),
        [LABELS[n] for n in S.CLASS_NAMES],
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.set_ylabel("Share of biopsy's\nEE cells")
    S.status_legend(ax, "below", ncols=2, columnspacing=0.8)


def makeFigure():
    return common.standalone_fixed(__name__)
