"""How many of the model's ligands each ligand-target prior covers: SecAct
signatures exist only for secreted proteins, so membrane-bound ligands
(CD70, CD80, ICAM1, ...) are unscorable with it."""

import numpy as np

from .. import common, style
from . import _common

GROUPS = {
    "Both": (True, True),
    "NicheNet only": (True, False),
    "SecAct only": (False, True),
    "Neither": (False, False),
}


def plot(fig):
    out = _common.load().query("universe == 'own'")
    covered = (
        out.groupby(["prior", "ligand"])["score"]
        .apply(lambda s: s.notna().any())
        .unstack(0)
    )
    counts = [
        int(((covered["nichenet"] == n) & (covered["secact"] == s)).sum())
        for n, s in GROUPS.values()
    ]
    ax = fig.subplots()
    y = np.arange(len(GROUPS))
    ax.barh(y, counts, color=style.COLORS["emphasis"], height=0.6)
    for yi, c in zip(y, counts, strict=True):
        ax.text(c + 2, yi, str(c), va="center", fontsize=style.FONT_SIZE_DENSE)
    ax.set_yticks(y, list(GROUPS))
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.2)
    ax.set_xlabel("Model ligands")
    ax.set_title(f"Prior coverage of {len(covered)} model ligands")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"])
