"""
Figure A7c: Single-cell CD70→CD27 communication scores in the independent
ALAD airway brushing dataset. Recovered vs Declined.
"""

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from .common import getSetup, subplotLabel


def makeFigure():
    ax, f = getSetup((8, 4), (1, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/alad.brush.sc.h5ad")
    alad_rename = {"ALAD int": "ALAD Int", "ALAD recovered": "Recovered", "ALAD declined": "Declined"}
    order = ["Recovered", "Declined"]

    # (a) All T cells as senders
    df_all = _comm_scores_by_patient(X, cd70_pos_only=False)
    df_all["ALAD"] = df_all["ALAD"].replace(alad_rename)
    df_all = df_all[df_all["ALAD"].isin(order)]
    print("All T-cell senders (Recovered vs Declined):")
    print(df_all.to_string())

    _boxstrip(ax[0], df_all, order, "Single-Cell CCC: CD70→CD27")

    # (b) CD70+ senders only
    df_pos = _comm_scores_by_patient(X, cd70_pos_only=True)
    df_pos["ALAD"] = df_pos["ALAD"].replace(alad_rename)
    df_pos = df_pos[df_pos["ALAD"].isin(order)]
    print("\nCD70+ senders only (Recovered vs Declined):")
    print(df_pos.to_string())

    _boxstrip(ax[1], df_pos, order, "CCC: CD70+ Senders Only")

    return f


def _comm_scores_by_patient(X, cd70_pos_only=False):
    """Per-patient mean CD70×CD27 communication score.

    For each patient: mean(CD70 in senders) × mean(CD27 in receivers),
    which equals the mean of the full sender×receiver outer product.
    """
    records = []
    for pid in X.obs["subj"].unique():
        patient = X[X.obs["subj"] == pid]
        senders = patient[patient.obs["cell_type_inintegrated"] == "Tcell"]
        receivers = patient[patient.obs["cell_type_inintegrated"] != "Tcell"]
        if senders.shape[0] == 0 or receivers.shape[0] == 0:
            continue

        lig = np.asarray(senders[:, "CD70"].X.todense()).ravel()
        rec = np.asarray(receivers[:, "CD27"].X.todense()).ravel()

        if cd70_pos_only:
            lig = lig[lig > 0]
            if len(lig) == 0:
                continue

        records.append({
            "patient": pid,
            "mean_comm_score": lig.mean() * rec.mean(),
            "ALAD": patient.obs["ALAD"].iloc[0],
            "n_sender": len(lig),
            "n_receiver": len(rec),
        })
    return pd.DataFrame(records)


def _boxstrip(ax, df, order, title):
    sns.boxplot(data=df, x="ALAD", y="mean_comm_score", order=order,
                boxprops=dict(facecolor="none"), ax=ax)
    sns.stripplot(data=df, x="ALAD", y="mean_comm_score", order=order,
                  size=6, jitter=0.15, ax=ax)
    ax.set_ylabel("Mean CD70×CD27\nCommunication Score")
    ax.set_xlabel("")
    ax.set_title(title)
