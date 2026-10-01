"""CP component stability under each score: aggregate FMS (median over
replicates) per resampling axis."""

import numpy as np
import pandas as pd

from ...validation.component_stability import STAGE_CPD, metric_interval
from .. import common, style
from . import _shared as S


def stability_table() -> pd.DataFrame:
    """(axis x method) median aggregate CP FMS."""
    medians = {}
    for method in S.ALL_SCORES:
        df = S.load_method_output(S.SPEC.stability_filename(), method)
        stats = metric_interval(df[df["Stage"] == STAGE_CPD], "FMS", by=("Axis",))
        medians[method] = stats.set_index("Axis")["Median"]
    table = pd.DataFrame(medians)
    return table.reindex([a for a in style.STABILITY_AXIS_COLORS if a in table.index])


def plot(fig):
    table = stability_table()
    ax = fig.subplots()
    S.grouped_bar(ax, table, "Aggregate CP FMS")
    ax.set_xticks(np.arange(len(table)), table.index, rotation=30, ha="right")
    ax.set_xlabel("Resampled")
    ax.set_ylim(0, 1.05)
    n = S.default_provenance("stability", "n_replicates")
    ax.set_title(f"CP stability ({n} replicates)")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"], legend=S.method_legend)
