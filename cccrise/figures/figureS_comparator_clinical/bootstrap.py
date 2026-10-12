"""Patient-resampled bootstrap of the 6-month outcome (4-way) association,
per component: the eta-squared interval (95%) and median of cccRISE
(``cccrise-anovaboot``) and of each CP comparator at its own coarse rank
(``cccrise-comparator-bootstrap``), with the full-cohort value (x) and each
method's permutation-null 95th percentile of the max eta-squared (dashes).
Right: the fraction of replicates with p < 0.05 and the median LR-mode match
score of the component."""

import numpy as np
import pandas as pd

from ...comparators.resampling import summarize_bootstrap_anova
from ...runners.datasets import VALIDATION_DIR
from ...runners.provenance import read_metadata
from .. import common, style
from . import _cccrise_boot as cccrise_boot
from ._common import SHORT, load

FILENAME = "comparator_anova_bootstrap_alad.parquet"
ASSOCIATION = "6-Month Outcome (4-Way)"


def _comparator_summaries() -> list[tuple[str, pd.DataFrame]]:
    replicates = pd.read_parquet(VALIDATION_DIR / FILENAME)
    references = read_metadata(VALIDATION_DIR / FILENAME)["references"]
    out = []
    for method, reps in replicates.groupby("Method", sort=False):
        ref = references[method]
        reference = pd.DataFrame(
            {
                "Component": np.arange(1, len(ref["reference_etasquared"]) + 1),
                "EtaSquared": ref["reference_etasquared"],
                "F": ref["reference_f"],
                "P": ref["reference_p"],
                "GroupMeanDiff": ref["reference_groupmeandiff"],
            }
        )
        out.append((str(method), summarize_bootstrap_anova(reps, reference)))
    return out


def _null_q95() -> dict[str, float]:
    null = load("anova_null", "coarse")
    null = null[null["Association"] == ASSOCIATION]
    return dict(zip(null["Method"], null["NullQ95"], strict=True))


def _draw_group(ax, table: pd.DataFrame, y0: float) -> list[float]:
    """One method's component intervals from row `y0`; returns their rows."""
    rows = []
    for i, (_, row) in enumerate(table.iterrows()):
        y = y0 + i
        color = (
            style.COLORS["emphasis"]
            if row["ReferenceP"] < 0.05
            else style.COLORS["ink"]
        )
        ax.plot([row["CILow"], row["CIHigh"]], [y, y], color=color, lw=1.5)
        ax.scatter(row["MedianEtaSquared"], y, color=color, s=10, zorder=3)
        ax.scatter(row["ReferenceEtaSquared"], y, color=color, marker="x", s=12)
        ax.text(
            1.02,
            y,
            f"{row['FracSignificant']:.0%}  {row['MedianMatchScore']:.2f}",
            transform=ax.get_yaxis_transform(),
            fontsize=style.FONT_SIZE_DENSE - 1,
            va="center",
        )
        rows.append(y)
    return rows


def plot(fig):
    if not (VALIDATION_DIR / FILENAME).exists():
        common.placeholder(fig, "comparator bootstrap not run yet")
        return
    groups = [("cccRISE", cccrise_boot.summary()), *_comparator_summaries()]
    q95 = _null_q95()
    ax = fig.subplots()
    ticks, labels, y0 = [], [], 0.0
    for method, table in groups:
        rows = _draw_group(ax, table, y0)
        ticks += rows
        labels += [f"{SHORT.get(method, method)} C{int(c)}" for c in table["Component"]]
        if method in q95:
            ax.plot(
                [q95[method]] * 2,
                [rows[0] - 0.4, rows[-1] + 0.4],
                color=style.COLORS["reference"],
                ls="--",
                lw=0.8,
            )
        y0 = rows[-1] + 1.6
    ax.set_yticks(ticks, labels, fontsize=style.FONT_SIZE_DENSE)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\eta^2$, 6-month outcome (bootstrap 95% interval)")
    ax.set_title("p<.05  match", loc="right", fontsize=style.FONT_SIZE_DENSE)
    ax.grid(False)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
