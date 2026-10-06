"""Does a prior reward the model's high-weight ligands, and how much of that
is the ligand's own expression in the receivers? Per setting: Spearman of
ligand weight vs score (ranked within component, pooled), raw and with both
residualized on the ligand's own pseudobulk log-TPM change between the
receiver groups; and Spearman of score vs that self-expression change.
SecAct's co-localization signatures largely contain the ligand's own
cell-of-origin program, so its scores track self-expression."""

import numpy as np
from scipy.stats import spearmanr

from .. import common, style
from . import _common

SERIES = {
    "weight vs score": ("weight", None, style.COLORS["emphasis"]),
    "weight vs score | self-expr.": (
        "weight",
        "ligand_log_tpm_diff",
        style.COLORS["highlight"],
    ),
    "self-expr. vs score": ("self", None, style.COLORS["reference"]),
}
OFFSET = 0.22
X_LIMITS = (-0.1, 0.4)


def plot(fig):
    own = _common.load().query("universe == 'own'")
    rows = _common.settings()
    ax = fig.subplots()
    for k, (label, (kind, control, color)) in enumerate(SERIES.items()):
        values = [_statistic(own, r, kind, control) for r in rows]
        y = np.arange(len(rows)) + (k - 1) * OFFSET
        ax.scatter(values, y, color=color, s=style.POINT_SIZE, label=label, zorder=3)
    common.reference_line(ax, 0, horizontal=False)
    ax.axhline(
        len(_common.SETTING_ORDER) - 0.5,
        color=style.COLORS["muted"],
        lw=style.LINE_WIDTH_THIN,
    )
    ax.set_yticks(range(len(rows)), [_common.setting_label(*r) for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ")
    ax.set_title("Weight–score association and self-expression")
    ax.set_xlim(*X_LIMITS)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.4, -0.22),
        ncols=3,
        fontsize=style.FONT_SIZE_DENSE,
    )


def _statistic(own, setting, kind, control) -> float:
    mask = (own[_common.SETTING] == setting).all(axis=1)
    s = own[mask].dropna(subset=["score", "ligand_log_tpm_diff"])
    if kind == "self":
        r = _common.within_component_ranks(s, ["score", "ligand_log_tpm_diff"])
        return float(spearmanr(r["score"], r["ligand_log_tpm_diff"])[0])
    return _common.weight_score_correlation(s, control)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
