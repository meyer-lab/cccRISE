"""CP rank selection by the multi-start rule (RANK_SELECTION_RULE.md), from
``python -m cccrise.runners.run_cpd_rank_rule`` on the canonical RISE fit.

Left: fit R2X by rank for every start, the production starts filled when
they pass the collinearity screen, random starts in grey (a diagnostic), the
best passing production fit as a line, and the R2X elbow as a diagnostic.
Right: the 10 production starts per rank stacked by screen outcome, the 9 of
10 line they must reach, core consistency's zone above (>= 80, 0-80, < 0;
the rank needs > 0), and a cross where the passing starts disagree. The
selected rank is the largest that passes."""

from ...runners.datasets import RANK_SELECTION_DIR
from ...runners.provenance import read_metadata
from .. import common

DATASET = "alad"
AXES = ("square", "square")


def plot(axs, dataset: str = DATASET):
    name = f"cpd_rank_rule_{dataset}.parquet"
    starts = common.load_runner_output(name, RANK_SELECTION_DIR)
    summary = common.load_runner_output(
        f"cpd_rank_rule_summary_{dataset}.parquet", RANK_SELECTION_DIR
    )
    run = read_metadata(RANK_SELECTION_DIR / name)
    rank = common.selected_ranks(dataset)[1]
    common.plot_rule_r2x(starts, summary, axs[0], rank, run["elbow_diagnostic"])
    common.plot_rule_screen(starts, summary, axs[1], rank, run["pass_share"])


def makeFigure():
    return common.standalone_fixed(__name__)
