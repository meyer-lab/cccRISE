"""The missed columns not explained by one sample, a marker or homophilic pair
or abundance: their 6-month outcome association (eta^2, nominal p on
read-depth-corrected profiles) and the smallest cccRISE CP rank, per RISE fit,
whose LR span contains them."""

import numpy as np

from .. import common, style
from . import _common

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    table = _common.audit()
    rows = table[table["class"] == "candidate biology"].sort_values(f"EtaSquared: {_common.OUTCOME}")
    eta = rows[f"EtaSquared: {_common.OUTCOME}"].to_numpy()
    p = rows[f"P: {_common.OUTCOME}"].to_numpy()
    y = np.arange(len(rows))
    ax.barh(y, eta, color=[style.COLORS["emphasis"] if v < 0.05 else style.COLORS["muted"] for v in p], lw=0)
    for yi, record in enumerate(rows.to_dict("records")):
        reach = [f"{label} CP {int(record[f'contained_from_rank: {tag}'])}"
                 for tag, label in _common.RISE_FITS.items()
                 if np.isfinite(record[f"contained_from_rank: {tag}"])]  # fmt: skip
        note = f"p={p[yi]:.3f}; " + (reach[0] if reach else "never contained")
        ax.text(eta[yi] + 0.005, yi, note, va="center", fontsize=style.FONT_SIZE_DENSE)
    ax.set_yticks(y, [f"{_common.pair_label(r)} ({r.fit})" for r in rows.itertuples()])
    ax.set_xlim(0, max(eta) * 2.2)
    ax.set_xlabel("6-mo. outcome $\\eta^2$")


def makeFigure():
    return common.standalone_fixed(__name__)
