"""The IBD RISE condition factor (A), each component divided by its largest
magnitude (keeping sign), as the ALAD supplement draws ALAD's: biopsies
grouped by inflammation status (a color strip left of the matrix), and
within a status ordered by their most heavily loaded component, then by that
loading."""

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from ... import common
from ...figure2.figureS_rise_factor_matrices._common import XLABEL, normalized_table
from .. import _common as S
from ._common import component_ticks

AXES = ("rect",)
STRIP_WIDTH = 1.2 * common.style.MM
STRIP_GAP = 0.4 * common.style.MM


def plot(axs):
    (ax,) = axs
    factor = S.factors()["pf2_A"]
    conditions = factor.values.astype(float)
    names = factor["condition"].values.astype(str)
    status = (
        S.samples().set_index(S.SPEC.condition_column)[S.GROUP_COLUMN].loc[names]
    ).to_numpy()
    rank = np.array([list(S.GROUP_LABELS).index(s) for s in status])
    order = common.dominant_order(conditions)
    order = order[np.argsort(rank[order], kind="stable")]
    common.plot_factor_heatmap(ax, conditions[order], XLABEL)
    component_ticks(ax, conditions.shape[1])
    common.source_data(
        "rise_condition_factor",
        normalized_table(names[order], conditions[order], "Sample").assign(
            Status=status[order]
        ),
        "RISE condition factor (A): each terminal ileum biopsy's (row, top to "
        "bottom as plotted) loading on each RISE component, divided by the "
        "component's largest |loading| (sign kept), with its inflammation status.",
    )
    ax.set_yticks([])
    status_strip(ax, status[order], f"Conditions (n={len(conditions)})")


def status_strip(ax, status: np.ndarray, ylabel: str) -> None:
    """A strip of each row's inflammation status left of the matrix, labeled
    `ylabel`, and its legend right of the matrix."""
    inches = ax.bbox.width / ax.figure.dpi
    strip = ax.inset_axes(
        (-(STRIP_WIDTH + STRIP_GAP) / inches, 0, STRIP_WIDTH / inches, 1)
    )
    rgb = [to_rgb(S.GROUP_COLORS[s]) for s in status]
    strip.imshow(np.array(rgb)[:, None], aspect="auto", interpolation="nearest")
    strip.set(xticks=[], yticks=[], ylabel=ylabel)
    strip.spines[:].set_visible(False)
    key = common.legend(
        ax,
        "right",
        handles=[Patch(color="none", label="Inflammation status")]
        + [Patch(color=S.GROUP_COLORS[g], label=n) for g, n in S.GROUP_LABELS.items()],
        handlelength=0.8,
    )
    key.get_texts()[0].set_fontweight("bold")


def makeFigure():
    return common.standalone_fixed(__name__)
