"""
Figure A3n-o: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
import pandas as pd
import seaborn as sns

from ..utils import (
    add_obs_cmp_both_label,
    add_obs_cmp_unique_two,
    aggregate_communication_scores_by_group,
)
from .common import (
    COVID19_PATH,
    getSetup,
    subplotLabel,
)


def makeFigure():
    ax, f = getSetup((10, 10), (3, 3))
    subplotLabel(ax)

    X = anndata.read_h5ad(COVID19_PATH)
    ccc_rise_cmp1 = 3
    ccc_rise_cmp2 = 5

    X_epi_sender = X[X.obs["celltype"] == "Epithelial"]
    X_epi_sender = add_obs_cmp_both_label(
        X_epi_sender,
        cmp1=ccc_rise_cmp1,
        cmp2=ccc_rise_cmp2,
        pos1=True,
        pos2=True,
        top_perc=10,
        type="sender",
    )
    X_epi_sender = add_obs_cmp_unique_two(
        X_epi_sender, cmp1=ccc_rise_cmp1, cmp2=ccc_rise_cmp2
    )
    X_epi_sender = X_epi_sender[X_epi_sender.obs["Label"] != "NoLabel"]

    X_epi_receiver = X[(X.obs["celltype"] == "Epithelial")]
    X_epi_receiver = add_obs_cmp_both_label(
        X_epi_receiver,
        cmp1=ccc_rise_cmp1,
        cmp2=ccc_rise_cmp2,
        pos1=True,
        pos2=True,
        top_perc=10,
        type="receiver",
    )
    X_epi_receiver = add_obs_cmp_unique_two(
        X_epi_receiver, cmp1=ccc_rise_cmp1, cmp2=ccc_rise_cmp2
    )
    X_epi_receiver = X_epi_receiver[X_epi_receiver.obs["Label"] != "NoLabel"]

    pairs = [
        ["PTN", "PTPRZ1"],
        ["PTN", "SDC1"],
        ["COL4A5", "SDC1"],
        ["CDH1", "CDH1"],
        ["OCLN", "OCLN"],
        ["PRSS3", "F2RL1"],
    ]

    communication_data = []
    for lig, rec in pairs:
        pair_name = f"{lig}-{rec}"

        # Calculate average communication score for each sender-receiver label combination
        pair_df = aggregate_communication_scores_by_group(
            X_epi_sender,
            X_epi_receiver,
            lig,
            rec,
            sender_group=X_epi_sender.obs["Label"].values,
            receiver_group=X_epi_receiver.obs["Label"].values,
        )
        pair_df = pair_df.rename(
            columns={"sender_group": "sender_label", "receiver_group": "receiver_label"}
        )
        pair_df["pair"] = pair_name
        pair_df["label_combination"] = (
            pair_df["sender_label"] + "→" + pair_df["receiver_label"]
        )
        communication_data.append(pair_df)

    # Convert to DataFrame
    comm_df = pd.concat(communication_data, ignore_index=True)

    # Create separate heatmap for each ligand-receptor pair
    for i, pair_name in enumerate(pairs):
        pair_label = f"{pair_name[0]}-{pair_name[1]}"

        # Filter data for this specific pair
        pair_data = comm_df[comm_df["pair"] == pair_label]

        if len(pair_data) == 0:
            print(f"No data found for pair: {pair_label}")
            continue

        # Create pivot table for this pair
        pivot_data = pair_data.pivot_table(
            values="communication_score",
            index="sender_label",
            columns="receiver_label",
            aggfunc="mean",
        )
        pivot_data = (pivot_data - pivot_data.min().min()) / (
            pivot_data.max().max() - pivot_data.min().min()
        )
        sns.heatmap(
            pivot_data,
            cmap="Purples",
            cbar_kws={"label": "Avg Communication Score"},
            ax=ax[i],
        )
        ax[i].set_xlabel("Receiver Label")
        ax[i].set_ylabel("Sender Label")

    return f
