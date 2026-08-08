"""
Figure A3f: Heatmaps of ligand-receptor expression products for selected cell types and components in CCC-RISE on BALF COVID-19 data.
"""

import anndata
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils import (
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
    aggregate_communication_scores_by_group,
)
from .common import (
    COVID19_PATH,
    getSetup,
    subplotLabel,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad(COVID19_PATH)
    ccc_rise_cmp = 3

    X_epi_sender = X[X.obs["celltype"] == "Epithelial"]
    X_epi_sender = add_obs_cmp_label(
        X_epi_sender, cmp=ccc_rise_cmp, pos=True, top_perc=10, type="sender"
    )
    X_epi_sender = add_obs_cmp_unique_one(X_epi_sender, cmp=ccc_rise_cmp)
    X_epi_sender = X_epi_sender[
        np.argsort(-X_epi_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    X_epi_receiver = X[(X.obs["celltype"] == "Epithelial")]
    X_epi_receiver = add_obs_cmp_label(
        X_epi_receiver, cmp=ccc_rise_cmp, pos=True, top_perc=10, type="receiver"
    )
    X_epi_receiver = add_obs_cmp_unique_one(X_epi_receiver, cmp=ccc_rise_cmp)

    pairs = [
        ["PTN", "PTPRZ1"],
        ["PTN", "SDC1"],
        ["COL4A5", "SDC1"],
        ["MDK", "PTPRZ1"],
        ["MDK", "SDC1"],
    ]

    # Calculate communication scores for each pair per sample
    communication_data = []

    for ligand, receptor in pairs:
        pair_name = f"{ligand}-{receptor}"

        # For each sample, calculate average communication score for no label and
        # labeled sender cells only
        for sample in X.obs["sample"].unique():
            # Get condition for this sample
            sample_condition = X.obs[X.obs["sample"] == sample]["condition"].iloc[0]

            sender_sample = X_epi_sender[X_epi_sender.obs["sample"] == sample]
            receiver_sample = X_epi_receiver[X_epi_receiver.obs["sample"] == sample]

            if sender_sample.n_obs == 0 or receiver_sample.n_obs == 0:
                continue

            sender_label_group = np.where(
                sender_sample.obs["Label"] == "NoLabel", "NoLabel", "Labeled"
            )
            scores = aggregate_communication_scores_by_group(
                sender_sample,
                receiver_sample,
                ligand,
                receptor,
                sender_group=sender_label_group,
            ).set_index("sender_group")["communication_score"]

            communication_data.append(
                {
                    "pair": pair_name,
                    "sample": sample,
                    "condition": sample_condition,
                    "communication_score_no_label": scores.get("NoLabel", np.nan),
                    "communication_score_label": scores.get("Labeled", np.nan),
                }
            )

    # Convert to DataFrame
    comm_df = pd.DataFrame(communication_data)

    # Plot both on the same plot for comparison with different names for legend with different colors for labeled vs no label. Combine into one plot and combine condition with label/no label
    comm_df_melted = pd.melt(
        comm_df,
        id_vars=["pair", "sample", "condition"],
        value_vars=["communication_score_no_label", "communication_score_label"],
        var_name="label_type",
        value_name="communication_score",
    )
    comm_df_melted["label_type"] = comm_df_melted["label_type"].map(
        {
            "communication_score_no_label": "No Label",
            "communication_score_label": "Labeled",
        }
    )
    comm_df_melted["condition_label"] = (
        comm_df_melted["condition"] + " - " + comm_df_melted["label_type"]
    )
    sns.boxplot(
        data=comm_df_melted,
        x="pair",
        y="communication_score",
        hue="condition_label",
        ax=ax[2],
    )
    ax[2].set_title(
        "Average Communication Scores by L-R Pair, Condition, and Label Type"
    )
    ax[2].set_xlabel("Ligand-Receptor Pair")
    ax[2].set_ylabel("Average Communication Score")
    ax[2].tick_params(axis="x", rotation=45)

    return f
