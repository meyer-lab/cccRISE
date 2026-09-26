"""
cccRISE ANOVA benchmark: the patient-status ANOVA heatmap of `figureAnova`,
drawn once per method and comparator rank mode.

Plots `benchmark_anova_<dataset>.csv`, written by
`uv run cccrise-benchmark --dataset <dataset>`. Rows are the comparator rank
modes (each method's own rank selection, then one shared rank); cccRISE's
published model is repeated in every row as the reference column. Cell
p-values are permutation p-values, family-wise across each method's components
(`benchmark_null`), when the CSV carries them.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .common import load_runner_csv, subplotLabel
from .figureAnova import plot_anova_summary_heatmap

SELECTED_DATASET = "alad"
SWEEP_CSV = f"benchmark_anova_{SELECTED_DATASET}.csv"
METHOD_ORDER = ("cccRISE", "Pseudobulk CPD", "Tensor-cell2cell", "STACCato")
RANK_MODE_LABELS = {"own": "own rank selection", "shared": "shared rank"}


def _panel_rows(df: pd.DataFrame) -> list[list[pd.DataFrame]]:
    """One row per comparator rank mode, one panel per method, cccRISE first."""
    reference = df[df["Method"] == "cccRISE"]
    rows = []
    for mode in RANK_MODE_LABELS:
        comparators = df[df["RankMode"] == mode]
        if comparators.empty:
            continue
        rows.append(
            [reference]
            + [
                comparators[comparators["Method"] == method]
                for method in METHOD_ORDER[1:]
                if (comparators["Method"] == method).any()
            ]
        )
    return rows


def makeFigure() -> Figure:
    df = load_runner_csv(SWEEP_CSV)
    rows = _panel_rows(df)
    n_cols = max(len(row) for row in rows)
    vmax = float(df["EtaSquared"].max())
    widths = [
        max(int(row[col]["Component"].max()) if col < len(row) else 1 for row in rows)
        for col in range(n_cols)
    ]

    f, axes = plt.subplots(
        len(rows),
        n_cols,
        figsize=(4.5 + 1.1 * sum(widths), 6.5 * len(rows)),
        gridspec_kw={"width_ratios": [w + 2 for w in widths]},
        layout="constrained",
        squeeze=False,
    )
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(row):
                ax.axis("off")
                continue
            panel = row[c]
            if "PermPAdj" in panel:
                panel = panel.assign(PAdj=panel["PermPAdj"])
            method, rank = panel["Method"].iloc[0], panel["Rank"].iloc[0]
            mode = (
                "published"
                if method == "cccRISE"
                else RANK_MODE_LABELS[panel["RankMode"].iloc[0]]
            )
            # One color scale across panels, so shading compares methods
            plot_anova_summary_heatmap(
                panel, ax, title=f"{method} ({mode}, rank {rank})", vmax=vmax
            )
            ax.set_xticklabels(
                [t.get_text().removeprefix("Component ") for t in ax.get_xticklabels()],
                rotation=0,
            )
            ax.set_xlabel("Component")
            if c > 0:
                ax.set_yticklabels([])

    subplotLabel([ax for ax in axes.flat if ax.axison])
    return f
