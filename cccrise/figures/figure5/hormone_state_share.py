"""Per biopsy (at least `MIN_EE` EE cells), the share of its EE cells in the
villus-like (C3-high) and crypt-like (C4-high) hormone states
(`_common.hormone_states`), by inflammation status (Kruskal-Wallis p)."""

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from .. import common, style
from . import _common as S

AXES = ("rect",)
MIN_EE = 3


def state_shares() -> pd.DataFrame:
    cells = S.programs("enteroendocrine_cells")
    state = S.hormone_states(cells)
    ee = state.notna()
    shares = pd.crosstab(cells.loc[ee, "Biopsy"], state[ee], normalize="index")
    counts = cells.loc[ee, "Biopsy"].value_counts()
    shares = shares[counts.reindex(shares.index) >= MIN_EE]
    status = S.samples().set_index(S.SPEC.condition_column)[S.GROUP_COLUMN]
    return shares.assign(Status=status.reindex(shares.index).to_numpy())


def plot(axs):
    (ax,) = axs
    shares = state_shares()
    common.source_data(
        "hormone_state_share_by_biopsy",
        shares.rename_axis("Biopsy").reset_index(),
        f"Share of each biopsy's EE cells (biopsies with at least {MIN_EE}) in "
        "each hormone state, with the biopsy's inflammation status; Kruskal-Wallis "
        "p across statuses above each state.",
    )
    rng = np.random.default_rng(0)
    for x, k in enumerate((3, 4)):
        groups = [
            shares.loc[shares["Status"] == g, S.STATE_NAMES[k]] for g in S.GROUP_LABELS
        ]
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
    ax.set_xticks([0, 1], [S.STATE_NAMES[k].replace(" (", "\n(") for k in (3, 4)])
    ax.set_ylabel("Share of biopsy's\nEE cells")
    S.status_legend(ax)


def makeFigure():
    return common.standalone_fixed(__name__)
