"""
Figure A7b: CD70 sender cell analysis in the independent ALAD airway
brushing dataset. CD70 expression across cell types and CD70+ T cell
abundance by ALAD group.
"""

import numpy as np
import pandas as pd
import seaborn as sns

from ..import_data import import_alad_brush
from .common import getSetup, subplotLabel


def makeFigure():
    ax, f = getSetup((10, 4), (1, 2))
    subplotLabel(ax)

    X = import_alad_brush()
    alad_rename = {
        "ALAD int": "ALAD Int",
        "ALAD recovered": "Recovered",
        "ALAD declined": "Declined",
    }
    alad_order = ["Control", "ALAD Int", "Recovered", "Declined"]

    # (a) CD70 expression by cell type
    ct_stats = []
    for ct in X.obs["cell_type_inintegrated"].unique():
        expr = np.asarray(
            X[X.obs["cell_type_inintegrated"] == ct, "CD70"].X.todense()
        ).ravel()
        ct_stats.append(
            {
                "cell_type": ct,
                "pct_CD70_pos": (expr > 0).mean() * 100,
                "n_cells": len(expr),
            }
        )
    ct_df = pd.DataFrame(ct_stats).sort_values("pct_CD70_pos")
    print("CD70 expression by cell type:")
    print(ct_df.to_string())

    ct_df.plot.barh(x="cell_type", y="pct_CD70_pos", ax=ax[0], legend=False)
    ax[0].set_xlabel("% CD70+ Cells")
    ax[0].set_ylabel("")
    ax[0].set_title("CD70 Expression by Cell Type")

    # (b) Per-patient % CD70+ T cells by ALAD
    X_t = X[X.obs["cell_type_inintegrated"] == "Tcell"]
    records = []
    for pid in X_t.obs["subj"].unique():
        mask = (X_t.obs["subj"] == pid).values
        expr = np.asarray(X_t[mask, "CD70"].X.todense()).ravel()
        records.append(
            {
                "patient": pid,
                "pct_CD70_pos": (expr > 0).mean() * 100,
                "ALAD": X_t.obs.loc[mask, "ALAD"].iloc[0],
                "n_cells": mask.sum(),
            }
        )
    df_b = pd.DataFrame(records)
    df_b["ALAD"] = df_b["ALAD"].replace(alad_rename)
    print("\nPer-patient % CD70+ T cells:")
    print(df_b.to_string())

    _boxstrip(
        ax[1],
        df_b,
        "pct_CD70_pos",
        alad_order,
        "% CD70+ T Cells",
        "CD70+ T Cells by ALAD",
    )

    return f


def _boxstrip(ax, df, y_col, order, ylabel, title):
    sns.boxplot(
        data=df,
        x="ALAD",
        y=y_col,
        order=order,
        boxprops={"facecolor": "none"},
        ax=ax,
    )
    sns.stripplot(data=df, x="ALAD", y=y_col, order=order, size=5, jitter=0.15, ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(title)
