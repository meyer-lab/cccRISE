"""
Figure A7b: CD70 expression in CD4+ T cells by ALAD status in independent
airway brushing dataset. Per-patient % of CD70+ CD4+ T cells across all
ALAD groups.
"""

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from .common import getSetup, subplotLabel


def makeFigure():
    ax, f = getSetup((3, 3), (1, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/alad.brush.sc.h5ad")
    X_tcell = X[X.obs["cell_type_inintegrated"] == "Tcell"]
    print(f"CD4+ T cells: {X_tcell.shape[0]} cells")

    gene_expr = np.asarray(X_tcell[:, "CD70"].X.todense()).ravel()
    is_positive = gene_expr > 0

    records = []
    for pid in X_tcell.obs["subj"].unique():
        mask = (X_tcell.obs["subj"] == pid).values
        n_total = mask.sum()
        n_pos = (mask & is_positive).sum()
        pct = (n_pos / n_total) * 100 if n_total > 0 else 0.0
        alad = X_tcell.obs.loc[mask, "ALAD"].iloc[0]
        records.append({"patient": pid, "pct_CD70_pos": pct, "ALAD": alad, "n_cells": n_total})
    df = pd.DataFrame(records)
    df["ALAD"] = df["ALAD"].replace({"ALAD int": "ALAD Int", "ALAD recovered": "Recovered", "ALAD declined": "Declined"})
    print(df.to_string())

    order = ["Control", "ALAD Int", "Recovered", "Declined"]
    sns.boxplot(
        data=df,
        x="ALAD",
        y="pct_CD70_pos",
        order=order,
        boxprops=dict(facecolor="none"),
        ax=ax[0],
    )
    sns.stripplot(
        data=df,
        x="ALAD",
        y="pct_CD70_pos",
        order=order,
        size=5,
        jitter=0.15,
        ax=ax[0],
    )
    ax[0].set_ylabel("% CD70+ CD4+ T cells")
    ax[0].set_xlabel("")
    ax[0].set_title("CD70 Expression in CD4+ T Cells")

    return f
