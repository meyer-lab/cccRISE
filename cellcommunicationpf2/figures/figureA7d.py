"""
Figure A7d: CD70+ T cells are enriched for CAMK4 co-expression in the
independent ALAD airway brushing dataset, validating the CCC-RISE signal.
"""

import numpy as np
import seaborn as sns
from scipy.stats import fisher_exact
from .common import getSetup, subplotLabel
from ..import_data import import_alad_brush


def makeFigure():
    ax, f = getSetup((8, 4), (1, 2))
    subplotLabel(ax)

    X = import_alad_brush()

    # Subset to T cells
    X_t = X[X.obs["cell_type_inintegrated"] == "Tcell"]
    cd70 = np.asarray(X_t[:, "CD70"].X.todense()).ravel()
    camk4 = np.asarray(X_t[:, "CAMK4"].X.todense()).ravel()
    alad = X_t.obs["ALAD"].values

    # (a) All T cells
    _plot_panel(ax[0], cd70, camk4, "All T Cells")

    # (b) Declined only
    mask_d = alad == "ALAD declined"
    _plot_panel(ax[1], cd70[mask_d], camk4[mask_d], "Declined")

    return f


def _plot_panel(ax, cd70, camk4, title):
    """Bar chart: % CAMK4+ in CD70- vs CD70+ T cells with Fisher's exact."""
    colors = sns.color_palette("colorblind", 2)

    cd70_pos = cd70 > 0
    camk4_pos = camk4 > 0
    table = [
        [(cd70_pos & camk4_pos).sum(), (cd70_pos & ~camk4_pos).sum()],
        [(~cd70_pos & camk4_pos).sum(), (~cd70_pos & ~camk4_pos).sum()],
    ]
    odr, p = fisher_exact(table)

    n_cd70pos = sum(table[0])
    n_cd70neg = sum(table[1])
    pct_pos = table[0][0] / n_cd70pos * 100 if n_cd70pos > 0 else 0
    pct_neg = table[1][0] / n_cd70neg * 100 if n_cd70neg > 0 else 0
    print(f"{title}: CD70+ {pct_pos:.1f}% (n={n_cd70pos}), "
          f"CD70- {pct_neg:.1f}% (n={n_cd70neg}), OR={odr:.2f}, p={p:.4f}")

    groups = ["CD70−", "CD70+"]
    pcts = [pct_neg, pct_pos]
    bars = ax.bar(groups, pcts, color=[colors[0], colors[1]], width=0.5)

    # Value labels on bars
    for bar, pct, n in zip(bars, pcts, [n_cd70neg, n_cd70pos]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=7)

    # Significance bracket
    y_max = max(pcts) + 10
    ax.plot([0, 0, 1, 1], [y_max - 1, y_max, y_max, y_max - 1], lw=1, c="k")
    p_str = f"p = {p:.3g}" if p >= 0.001 else "p < 0.001"
    ax.text(0.5, y_max + 0.5, p_str, ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("% CAMK4+")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.set_ylim(0, y_max + 8)
