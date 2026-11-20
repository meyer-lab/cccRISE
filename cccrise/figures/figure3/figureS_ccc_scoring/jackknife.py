"""How stable each score's CP fit is under leaving one patient out: each
jackknife refit's aggregate factor match score (mean over components, LR
mode) against the score's full-cohort fit, from
``uv run cccrise-altcommscore --jackknife``."""

import seaborn as sns

from ....runners.datasets import ALT_SCORE_METHODS_DIR
from ... import common, style
from . import _common as S

HINT = (
    "    uv run cccrise-altcommscore --dataset alad --rise-rank 17 --cp-rank 6 "
    "--jackknife"
)


def plot(fig):
    table = common.load_runner_output(
        S.SPEC.alt_score_jackknife_filename(), ALT_SCORE_METHODS_DIR, HINT
    )
    common.source_data(
        "jackknife_fms_by_score",
        table[["Method", "LeftOut", "FMS"]]
        .assign(Method=table["Method"].map(S.SCORE_LABELS))
        .rename(columns={"LeftOut": "Left-out patient", "FMS": "Aggregate FMS"}),
        "One row per (score, leave-one-patient-out refit): the mean over CPD "
        "components of the refit's factor match score (LR mode) to the score's "
        "full-cohort fit.",
    )
    ax = fig.subplots()
    kwargs = {"data": table, "x": "Method", "y": "FMS", "order": S.ALL_SCORES, "ax": ax}
    sns.boxplot(
        **kwargs,
        hue="Method",
        palette=S.SCORE_COLORS,
        legend=False,
        fliersize=0,
        linewidth=style.LINE_WIDTH_THIN,
    )
    common.stripplot(**kwargs, size=2, color="black")
    ax.set_xticks(range(len(S.ALL_SCORES)), S.METHOD_TICKS)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("")
    ax.set_ylabel("Jackknife aggregate FMS")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"], legend=S.method_legend)
