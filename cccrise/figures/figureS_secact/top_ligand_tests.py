"""Top ligands vs all other ligands (`nichenet.top_ligand_tests`) for every
prior x receiver profile x scoring setting and CP component, each prior's top
set drawn from its own scorable ligands. Color: signed -log10 p (red = top
ligands score higher); text: BH q across components."""

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from ...runners.run_prior_comparison import summarize
from .. import common, style
from . import _common

P_CAP = 3


def plot(fig):
    tests = summarize(_common.load().query("universe == 'own'")).set_index(
        _common.SETTING + ["component"]
    )
    rows = _common.settings()
    components = sorted(tests.index.get_level_values("component").unique())
    signed = np.array(
        [[_signed_log_p(tests.loc[(*r, c)]) for c in components] for r in rows]
    )
    ax = fig.subplots()
    image = ax.imshow(
        signed,
        cmap=style.CMAPS["diverging"],
        norm=TwoSlopeNorm(0, -P_CAP, P_CAP),
        aspect="auto",
    )
    for i, r in enumerate(rows):
        for j, c in enumerate(components):
            q = tests.loc[(*r, c), "q"]
            ax.text(
                j,
                i,
                f"{q:.2g}",
                ha="center",
                va="center",
                fontsize=style.FONT_SIZE_DENSE,
                fontweight="bold" if q < common.SIG_ALPHA else "normal",
                color=style.COLORS["ink"],
            )
    ax.axhline(len(_common.SETTING_ORDER) - 0.5, color="white", lw=2)
    ax.set_xticks(range(len(components)), components)
    ax.set_yticks(range(len(rows)), [_common.setting_label(*r) for r in rows])
    ax.set_xlabel("CP component")
    ax.set_title("Top vs other ligands: BH q (bold < 0.05)")
    fig.colorbar(image, ax=ax, label="signed −log10 p", shrink=0.8, pad=0.02)


def _signed_log_p(test) -> float:
    sign = 1 if test["direction"] == "up" else -1
    return sign * min(-np.log10(test["p"]), P_CAP)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["two_thirds"])
