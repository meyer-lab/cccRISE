"""
Figure 3: cccRISE components resolve epithelial cell communication
programs associated with disease severity.

Panel letters follow the manuscript figure. All panels are generated from
data already computed and stored on the loaded AnnData object.

TODO(IBD): NEEDS CHANGING OR REMOVAL. This figure was written for the retired
BALF cohort, which the IBD cohort supersedes. Its data loader now points at
IBD, but the components (CMP1, CMP2), severity ranking, cell types and LR pairs
below were picked for the old cohort and do not carry over: re-pick them for
IBD, or delete this figure.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

from ..utils import (
    add_obs_cmp_both_label,
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
    add_obs_cmp_unique_two,
    aggregate_communication_scores_by_group,
)
from .common import get_ibd_data
from .commonFuncs.plotFactors import plot_interaction_heatmap, plot_pair_lr_factors
from .commonFuncs.plotGeneral import get_sample_to_group, symmetrize_axes
from .commonFuncs.plotPaCMAP import plot_wc_pacmap

CMP1 = 3
CMP2 = 5


def label_panel(ax, letter):
    ax.text(
        -0.25,
        1.2,
        letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=12,
        va="top",
    )


def _condition_colors(X, condition_column, group_col):
    sample_to_group = get_sample_to_group(X, condition_column, group_col)
    pal = sns.color_palette(palette="Set2", n_colors=len(sample_to_group.unique()))
    pal = pal.as_hex()
    color_map = dict(zip(sample_to_group.unique(), pal))
    colors = [
        color_map[sample_to_group.loc[condition]] for condition in sample_to_group.index
    ]
    return sample_to_group, colors


def _epithelial_weight_violin(X, cmp, ax_sender, ax_receiver):
    for ax, factor, title in (
        (ax_sender, "sc_B", "Sender"),
        (ax_receiver, "rc_C", "Receiver"),
    ):
        X_epi = X[X.obs["celltype"] == "Epithelial"].obsm[factor][:, cmp - 1]
        X_epi = X_epi[np.argsort(X_epi)]
        thresholds = np.percentile(X_epi, [10, 30, 50, 70, 90, 95, 99])
        bounds = [-np.inf, *thresholds, np.inf]
        labels = [
            "0-10%",
            "10-30%",
            "30-50%",
            "50-70%",
            "70-90%",
            "90-95%",
            "95-99%",
            "99-100%",
        ]
        groups = pd.cut(X_epi, bins=bounds, labels=labels)
        df = pd.DataFrame({"Epithelial Weight": X_epi, "Group": groups})

        sns.violinplot(
            data=df, ax=ax, x="Group", y="Epithelial Weight", order=labels, cut=0
        )
        ax.set_xlabel("Epithelial Weight Distribution")
        ax.set_ylabel(f"{title} Cell Component {cmp} Association")
        ax.tick_params(axis="x", rotation=45)


def makeFigure():
    X = get_ibd_data(project_cells=True, do_embedding=True)
    condition_column = "sample"
    group_col = "condition"

    f = plt.figure(figsize=(22, 18), layout="constrained")
    grid = f.subfigures(4, 1, squeeze=False, height_ratios=[1.0, 1.0, 1.0, 1.1])
    subfigs = grid[:, 0]

    # ------------------------------------------------------------------
    # Row 1: (a) severity correlation, (b/c) Cmp.3 sender/receiver PaCMAP,
    #        (d/e) Cmp.3 sender/receiver violin
    # ------------------------------------------------------------------
    row1 = subfigs[0].subplots(1, 5)
    ax_a, ax_b, ax_c, ax_d, ax_e = row1

    sample_to_group, _ = _condition_colors(X, condition_column, group_col)
    A_factor = X.uns["A"]
    severity_map = {k: i + 1 for i, k in enumerate(np.unique(sample_to_group))}
    severity_numeric = sample_to_group.map(severity_map).values
    cmp = A_factor.shape[1]
    spearman_corr = [spearmanr(A_factor[:, i], severity_numeric)[0] for i in range(cmp)]
    spearman_df = pd.DataFrame(
        {
            "Component": [f"{i + 1}" for i in range(cmp)],
            "Spearman Correlation": spearman_corr,
        }
    )
    sns.barplot(
        x="Component",
        y="Spearman Correlation",
        data=spearman_df,
        ax=ax_a,
        color="black",
    )
    ax_a.set_ylim(-0.1, 1)
    ax_a.set_title("Condition Factor Vs. Severity")
    label_panel(ax_a, "a")

    plot_wc_pacmap(X, CMP1, ax_b, cbarMax=0.3, factor_matrix="B")
    ax_b.set_title(f"Cmp. {CMP1}: Sender Cells")
    label_panel(ax_b, "b")

    plot_wc_pacmap(X, CMP1, ax_c, cbarMax=0.3, factor_matrix="C")
    ax_c.set_title(f"Cmp. {CMP1}: Receiver Cells")
    label_panel(ax_c, "c")

    _epithelial_weight_violin(X, CMP1, ax_d, ax_e)
    label_panel(ax_d, "d")
    label_panel(ax_e, "e")

    # ------------------------------------------------------------------
    # Row 2: (f) labeled-vs-rest communication scores, (g/h) Cmp.5
    #        sender/receiver PaCMAP, (i/j) Cmp.5 sender/receiver violin
    # ------------------------------------------------------------------
    row2 = subfigs[1].subplots(1, 5)
    ax_f, ax_g, ax_h, ax_i, ax_j = row2

    X_epi_sender = X[X.obs["celltype"] == "Epithelial"]
    X_epi_sender = add_obs_cmp_label(
        X_epi_sender, cmp=CMP1, pos=True, top_perc=10, type="sender"
    )
    X_epi_sender = add_obs_cmp_unique_one(X_epi_sender, cmp=CMP1)

    X_epi_receiver = X[X.obs["celltype"] == "Epithelial"]
    X_epi_receiver = add_obs_cmp_label(
        X_epi_receiver, cmp=CMP1, pos=True, top_perc=10, type="receiver"
    )

    pairs = [
        ["PTN", "PTPRZ1"],
        ["PTN", "SDC1"],
        ["COL4A5", "SDC1"],
        ["MDK", "PTPRZ1"],
        ["MDK", "SDC1"],
    ]
    communication_data = []
    for ligand, receptor in pairs:
        pair_name = f"{ligand}-{receptor}"
        for sample in X.obs["sample"].unique():
            sample_condition = X.obs[X.obs["sample"] == sample]["condition"].iloc[0]
            sender_sample = X_epi_sender[X_epi_sender.obs["sample"] == sample]
            receiver_sample = X_epi_receiver[X_epi_receiver.obs["sample"] == sample]
            if sender_sample.n_obs == 0 or receiver_sample.n_obs == 0:
                continue
            sender_label_group = np.where(
                sender_sample.obs["Label"] == "NoLabel", "0-90%", "Top 10%"
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
                    "0-90%": scores.get("0-90%", np.nan),
                    "Top 10%": scores.get("Top 10%", np.nan),
                }
            )
    comm_df = pd.DataFrame(communication_data)
    comm_df_melted = pd.melt(
        comm_df,
        id_vars=["pair", "sample", "condition"],
        value_vars=["0-90%", "Top 10%"],
        var_name="label_type",
        value_name="communication_score",
    )
    comm_df_melted["condition_label"] = (
        comm_df_melted["condition"] + ": " + comm_df_melted["label_type"]
    )
    sns.boxplot(
        data=comm_df_melted,
        x="pair",
        y="communication_score",
        hue="condition_label",
        ax=ax_f,
    )
    ax_f.set_xlabel("Ligand-Receptor Pair")
    ax_f.set_ylabel("Average Communication Score")
    ax_f.tick_params(axis="x", rotation=45)
    ax_f.legend(fontsize=6)
    label_panel(ax_f, "f")

    plot_wc_pacmap(X, CMP2, ax_g, cbarMax=0.3, factor_matrix="B")
    ax_g.set_title(f"Cmp. {CMP2}: Sender Cells")
    label_panel(ax_g, "g")

    plot_wc_pacmap(X, CMP2, ax_h, cbarMax=0.3, factor_matrix="C")
    ax_h.set_title(f"Cmp. {CMP2}: Receiver Cells")
    label_panel(ax_h, "h")

    X_epithelial = X[X.obs["celltype"] == "Epithelial"]
    sns.violinplot(data=X_epithelial.obsm["sc_B"][:, CMP2 - 1], ax=ax_i, cut=0)
    ax_i.set_ylim(-0.15, 0.5)
    ax_i.set_xlabel("Epithelial Cell Weight Distribution")
    ax_i.set_ylabel(f"Sender Cell Component {CMP2} Association")
    label_panel(ax_i, "i")

    sns.violinplot(data=X_epithelial.obsm["rc_C"][:, CMP2 - 1], ax=ax_j, cut=0)
    ax_j.set_ylim(-0.15, 0.5)
    ax_j.set_xlabel("Epithelial Cell Weight Distribution")
    ax_j.set_ylabel(f"Receiver Cell Component {CMP2} Association")
    label_panel(ax_j, "j")

    # ------------------------------------------------------------------
    # Row 3: (k) CDH1-CDH1 heatmap, (l) condition scatter, (m) LR scatter
    # ------------------------------------------------------------------
    row3 = subfigs[2].subplots(1, 3)
    ax_k, ax_l, ax_m = row3

    plot_interaction_heatmap(
        X,
        cmp=CMP2,
        ligand="CDH1",
        receptor="CDH1",
        ax=ax_k,
        sender_celltype="Epithelial",
        receiver_celltype="Epithelial",
        sender_label="Sender Epithelial Cells",
        receiver_label="Receiver Epithelial Cells",
    )
    label_panel(ax_k, "k")

    _, colors = _condition_colors(X, condition_column, group_col)
    ax_l.scatter(A_factor[:, CMP1 - 1], A_factor[:, CMP2 - 1], c=colors)
    ax_l.set_xscale("log")
    ax_l.set_yscale("log")
    ax_l.set_xlabel(f"cccRISE Condition Component {CMP1}")
    ax_l.set_ylabel(f"cccRISE Condition Component {CMP2}")
    label_panel(ax_l, "l")

    plot_pair_lr_factors(X, CMP1, CMP2, ax_m)
    symmetrize_axes(ax_m)
    ax_m.set_xlabel(f"cccRISE LR Cmp. {CMP1}")
    ax_m.set_ylabel(f"cccRISE LR Cmp. {CMP2}")
    label_panel(ax_m, "m")

    # ------------------------------------------------------------------
    # Row 4: (n) PTN-PTPRZ1 co-selection heatmap, (o) CDH1-CDH1 co-selection heatmap
    # ------------------------------------------------------------------
    row4 = subfigs[3].subplots(1, 2)
    ax_n, ax_o = row4

    X_epi_sender_both = X[X.obs["celltype"] == "Epithelial"]
    X_epi_sender_both = add_obs_cmp_both_label(
        X_epi_sender_both,
        cmp1=CMP1,
        cmp2=CMP2,
        pos1=True,
        pos2=True,
        top_perc=10,
        type="sender",
    )
    X_epi_sender_both = add_obs_cmp_unique_two(X_epi_sender_both, cmp1=CMP1, cmp2=CMP2)
    X_epi_sender_both = X_epi_sender_both[X_epi_sender_both.obs["Label"] != "NoLabel"]

    X_epi_receiver_both = X[X.obs["celltype"] == "Epithelial"]
    X_epi_receiver_both = add_obs_cmp_both_label(
        X_epi_receiver_both,
        cmp1=CMP1,
        cmp2=CMP2,
        pos1=True,
        pos2=True,
        top_perc=10,
        type="receiver",
    )
    X_epi_receiver_both = add_obs_cmp_unique_two(
        X_epi_receiver_both, cmp1=CMP1, cmp2=CMP2
    )
    X_epi_receiver_both = X_epi_receiver_both[
        X_epi_receiver_both.obs["Label"] != "NoLabel"
    ]

    no_pairs = [["PTN", "PTPRZ1"], ["CDH1", "CDH1"]]
    label_rename = {
        f"Cmp{CMP1}": f"Cmp. {CMP1}",
        f"Cmp{CMP2}": f"Cmp. {CMP2}",
        "Both": "Co-selected",
    }
    order = [f"Cmp. {CMP1}", f"Cmp. {CMP2}", "Co-selected"]
    for ax, (lig, rec) in zip((ax_n, ax_o), no_pairs):
        pair_df = aggregate_communication_scores_by_group(
            X_epi_sender_both,
            X_epi_receiver_both,
            lig,
            rec,
            sender_group=X_epi_sender_both.obs["Label"].map(label_rename).values,
            receiver_group=X_epi_receiver_both.obs["Label"].map(label_rename).values,
        )
        pivot = pair_df.pivot_table(
            values="communication_score",
            index="sender_group",
            columns="receiver_group",
            aggfunc="mean",
        ).reindex(index=order, columns=order)
        pivot = (pivot - pivot.min().min()) / (pivot.max().max() - pivot.min().min())
        sns.heatmap(
            pivot,
            cmap="Purples",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Avg Communication Score"},
        )
        ax.set_xlabel("Top 10% Sender/Receiver Cells")
        ax.set_ylabel("Top 10% Sender Epithelial Cells")
        ax.set_title(f"{lig}-{rec} Interaction")

    label_panel(ax_n, "n")
    label_panel(ax_o, "o")

    return f
