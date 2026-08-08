"""
Figure 4: CCC-RISE identifies communication programs distinguishing ALAD
lung transplant recipients from controls.

Panel letters follow the manuscript figure. Panel (a) is a hand-drawn
schematic with no corresponding plotting script and is left as a
placeholder. Panels (b) and (c) show 10-fold CV ROC AUC vs. RISE/CPD rank;
the underlying rank sweep (`rise_ranks_logreg`/`cpd_ranks_logreg` in
`logreg.py`) is slow (many tensor decompositions x repeated CV), so it is
included here but left commented out, matching the upstream scripts it was
assembled from.

Panels (k), (q), (r), (w), (x) show per-gene logistic regression weights
(`population_gene_logreg_weights` in `logreg.py`, fit over the full ~16k
gene panel with repeated 10-fold CV). This is also slow, so those calls are
left commented out below their placeholder axes; uncomment to run them.
"""

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ..logreg import ccc_rise_logreg_weights
from ..utils import (
    binned_complex_ligand_product_matrix,
    binned_expression_product_matrix,
    correct_conditions,
)
from .common import ALAD_PATH
from .commonFuncs.plotFactors import plot_lr_factors
from .commonFuncs.plotGeneral import binarize_group, get_sample_to_group
from .commonFuncs.plotPaCMAP import plot_labels_pacmap, plot_wc_pacmap

CMP_ALAD = 5
CMP_DC = 13
CMP_NK = 17
N_TOP_GENES = 20


def label_panel(ax, letter):
    ax.text(
        -0.3,
        1.2,
        letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=12,
        va="top",
    )


def _gene_weight_barplot(weights: pd.Series, ax, n=N_TOP_GENES):
    top = weights.iloc[:n]
    sns.barplot(x=top.values, y=top.index, ax=ax, color="goldenrod")
    ax.set_xlabel("Logistic Regression Weight")
    ax.set_ylabel("")


def _gene_weight_placeholder(ax, letter, title):
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"({letter}) {title}\n(per-gene logistic regression over ~16k\n"
        "genes is slow; see commented-out\ncode below)",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        transform=ax.transAxes,
    )
    label_panel(ax, letter)


def _ordered_population(X, celltype_col, celltype, cmp, factor_type):
    Xp = X[X.obs[celltype_col] == celltype]
    if factor_type == "sender":
        return Xp[np.argsort(-Xp.obsm["sc_B"][:, cmp - 1])]
    return Xp[np.argsort(Xp.obsm["rc_C"][:, cmp - 1])]


def makeFigure():
    X = anndata.read_h5ad(ALAD_PATH)
    celltype_col = "broad_cell_type"

    f = plt.figure(figsize=(24, 26), layout="constrained")
    subfigs = f.subfigures(5, 1)

    # ------------------------------------------------------------------
    # Row 1: (a) schematic, (b/c) rank-vs-AUC placeholders,
    #        (d) component logreg weights, (e) LR pair factor heatmap
    # ------------------------------------------------------------------
    row1 = subfigs[0].subplots(1, 5, width_ratios=[1.3, 1, 1, 1, 2])
    ax_a, ax_b, ax_c, ax_d, ax_e = row1

    ax_a.axis("off")
    ax_a.text(
        0.5,
        0.5,
        "(a) Schematic\n(no plotting script;\nsee manuscript figure)",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        transform=ax_a.transAxes,
    )
    label_panel(ax_a, "a")

    for ax, letter, xlabel in ((ax_b, "b", "RISE Rank"), (ax_c, "c", "CPD Rank")):
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"({letter}) 10-Fold CV ALAD ROC AUC\nvs. {xlabel}\n"
            "(rank sweep is slow; see\ncommented-out code below)",
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
            transform=ax.transAxes,
        )
        label_panel(ax, letter)

    # --- Panels b/c: rank sweep, left commented out (slow: many RISE/CPD
    # decompositions x RepeatedStratifiedKFold(10, 5) logistic regressions) ---
    # from ..import_data import add_cond_idxs, import_alad, import_ligand_receptor_pairs
    # from ..logreg import rise_ranks_logreg, cpd_ranks_logreg
    # from ..tensor import calculate_interaction_tensor
    #
    # X_raw = import_alad(gene_threshold=0.01, normalize=True)
    # lr_pairs = import_ligand_receptor_pairs()
    # condition_column = "dsco_id"
    # X_raw = add_cond_idxs(X_raw, condition_column)
    # group_col = "ALADstatus"
    # sample_to_group = get_sample_to_group(X_raw, condition_column, group_col)
    # sample_to_group = binarize_group(sample_to_group, "control", "control", "alad")
    #
    # rise_rank_list = list(np.append([1], range(5, 66, 5)))
    # scores_aucroc_rise, _ = rise_ranks_logreg(
    #     X_raw, rise_rank_list, sample_to_group, n_iter_max=10000, tolerance=1e-9
    # )
    # ax_b.plot(rise_rank_list, scores_aucroc_rise)
    #
    # interaction_tensor = calculate_interaction_tensor(X_raw, lr_pairs, rise_rank=15)
    # cpd_rank_list = list(range(2, 42, 2))
    # scores_aucroc_cpd, _ = cpd_ranks_logreg(
    #     X_raw, interaction_tensor, cpd_rank_list, sample_to_group, n_iter_max=10000
    # )
    # ax_c.plot(cpd_rank_list, scores_aucroc_cpd)

    condition_column = "dsco_id"
    X.uns["A"] = correct_conditions(X)
    group_col = "ALADstatus"
    sample_to_group_bin = get_sample_to_group(X, condition_column, group_col)
    sample_to_group_bin = binarize_group(
        sample_to_group_bin, "control", "control", "alad"
    )

    auc_roc_weights, _ = ccc_rise_logreg_weights(X, sample_to_group_bin)
    ax_d.bar(np.arange(len(auc_roc_weights)) + 1, auc_roc_weights)
    ax_d.set_xlabel("Component")
    ax_d.set_ylabel("Logistic Regression Weight")
    ym = np.max(np.abs(auc_roc_weights)) * 1.1
    ax_d.set_ylim(-ym, ym)
    label_panel(ax_d, "d")

    plot_lr_factors(X, ax_e, trim=True, weight=0.06)
    ax_e.set_title("Ligand-Receptor Pair Factor")
    label_panel(ax_e, "e")

    # ------------------------------------------------------------------
    # Row 2: (f) cell type PaCMAP, (g/h) Cmp.5 sender/receiver PaCMAP,
    #        (i) CD40LG-CD40, (j) LTA&LTB-LTBR
    # ------------------------------------------------------------------
    row2 = subfigs[1].subplots(1, 5)
    ax_f, ax_g, ax_h, ax_i, ax_j = row2

    pal = sns.color_palette("Set3").as_hex()
    plot_labels_pacmap(X, labelType=celltype_col, ax=ax_f, color_key=pal)
    ax_f.set_title("Projection Embedding")
    label_panel(ax_f, "f")

    plot_wc_pacmap(X, CMP_ALAD, ax_g, cbarMax=0.3, factor_matrix="B")
    ax_g.set_title(f"Cmp. {CMP_ALAD}: Sender Cells")
    label_panel(ax_g, "g")

    plot_wc_pacmap(X, CMP_ALAD, ax_h, cbarMax=0.3, factor_matrix="C")
    ax_h.set_title(f"Cmp. {CMP_ALAD}: Receiver Cells")
    label_panel(ax_h, "h")

    cd4t_sender = _ordered_population(
        X, celltype_col, "CD4 T cells", CMP_ALAD, "sender"
    )
    dc_receiver = _ordered_population(
        X, celltype_col, "Dendritic Cells", CMP_ALAD, "receiver"
    )

    df = binned_expression_product_matrix(cd4t_sender, dc_receiver, "CD40LG", "CD40")
    sns.heatmap(df, ax=ax_i, cmap="rocket")
    ax_i.set_title("CD40LG-CD40 Interaction")
    ax_i.set_xlabel("Receiver DC Population")
    ax_i.set_ylabel("Sender CD4 T cell Population")
    ax_i.set_xticks([])
    ax_i.set_yticks([])
    label_panel(ax_i, "i")

    df = binned_complex_ligand_product_matrix(
        cd4t_sender, dc_receiver, ["LTA", "LTB"], "LTBR"
    )
    sns.heatmap(df, ax=ax_j, cmap="rocket")
    ax_j.set_title("LTA&LTB-LTBR Interaction")
    ax_j.set_xlabel("Receiver DC Population")
    ax_j.set_ylabel("Sender CD4 T cell Population")
    ax_j.set_xticks([])
    ax_j.set_yticks([])
    label_panel(ax_j, "j")

    # ------------------------------------------------------------------
    # Row 3: (k) Cmp.5 gene weights (CD4 T, bottom 10%),
    #        (l/m) Cmp.13 sender/receiver PaCMAP, (n) Cmp.13 DC violin
    # ------------------------------------------------------------------
    row3 = subfigs[2].subplots(1, 4, width_ratios=[1.6, 1, 1, 1.6])
    ax_k, ax_l, ax_m, ax_n = row3

    _gene_weight_placeholder(ax_k, "k", "Bottom 10% Vs. Rest of Sender CD4 T cells")
    # from ..logreg import population_gene_logreg_weights
    # cd4t_weights = population_gene_logreg_weights(
    #     X[X.obs[celltype_col] == "CD4 T cells"], CMP_ALAD, "sender", pos=False
    # )
    # _gene_weight_barplot(cd4t_weights.sort_values(ascending=False), ax_k)
    # ax_k.set_title("Bottom 10% Vs. Rest of Sender CD4 T cells")
    # ax_k.set_xlabel(f"Cmp. {CMP_ALAD} Logistic Regression Weights")
    # label_panel(ax_k, "k")

    plot_wc_pacmap(X, CMP_DC, ax_l, cbarMax=0.3, factor_matrix="B")
    ax_l.set_title(f"Cmp. {CMP_DC}: Sender Cells")
    label_panel(ax_l, "l")

    plot_wc_pacmap(X, CMP_DC, ax_m, cbarMax=0.3, factor_matrix="C")
    ax_m.set_title(f"Cmp. {CMP_DC}: Receiver Cells")
    label_panel(ax_m, "m")

    dc_sender_13 = X[X.obs[celltype_col] == "Dendritic Cells"].obsm["sc_B"][
        :, CMP_DC - 1
    ]
    sns.violinplot(data=dc_sender_13, ax=ax_n, cut=0)
    ax_n.set_xlabel("DCs Weight Distribution")
    ax_n.set_ylabel(f"Sender Cell Component {CMP_DC} Association")
    label_panel(ax_n, "n")

    # ------------------------------------------------------------------
    # Row 4: (o) Cmp.13 epithelial receiver violin, (p) CD200-CD200R1,
    #        (q/r) Cmp.13 gene weights (DC sender, Epithelial receiver)
    # ------------------------------------------------------------------
    row4 = subfigs[3].subplots(1, 4, width_ratios=[1, 1.3, 1.6, 1.6])
    ax_o, ax_p, ax_q, ax_r = row4

    epi_receiver_13 = X[X.obs[celltype_col] == "Epithelial cells"].obsm["rc_C"][
        :, CMP_DC - 1
    ]
    sns.violinplot(data=epi_receiver_13, ax=ax_o, cut=0)
    ax_o.set_xlabel("Epithelial Cells Weight Distribution")
    ax_o.set_ylabel(f"Receiver Cell Component {CMP_DC} Association")
    label_panel(ax_o, "o")

    dc_sender = _ordered_population(
        X, celltype_col, "Dendritic Cells", CMP_DC, "sender"
    )
    epi_receiver = _ordered_population(
        X, celltype_col, "Epithelial cells", CMP_DC, "receiver"
    )
    df = binned_expression_product_matrix(dc_sender, epi_receiver, "CD200", "CD200R1")
    sns.heatmap(df, ax=ax_p, cmap="rocket", vmax=0.015)
    ax_p.set_title("CD200-CD200R1 Interaction")
    ax_p.set_xlabel("Receiver Epithelial Cell Population")
    ax_p.set_ylabel("Sender DC Population")
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    label_panel(ax_p, "p")

    _gene_weight_placeholder(ax_q, "q", "Top 10% Vs. Rest of Sender DCs")
    # from ..logreg import population_gene_logreg_weights
    # dc_weights = population_gene_logreg_weights(
    #     X[X.obs[celltype_col] == "Dendritic Cells"], CMP_DC, "sender", pos=True
    # )
    # _gene_weight_barplot(dc_weights, ax_q)
    # ax_q.set_title("Top 10% Vs. Rest of Sender DCs")
    # ax_q.set_xlabel(f"Cmp. {CMP_DC} Logistic Regression Weights")
    # label_panel(ax_q, "q")

    _gene_weight_placeholder(ax_r, "r", "Top 10% Vs. Rest of Receiver Epithelial Cells")
    # epi_weights = population_gene_logreg_weights(
    #     X[X.obs[celltype_col] == "Epithelial cells"], CMP_DC, "receiver", pos=True
    # )
    # _gene_weight_barplot(epi_weights, ax_r)
    # ax_r.set_title("Top 10% Vs. Rest of Receiver Epithelial Cells")
    # ax_r.set_xlabel(f"Cmp. {CMP_DC} Logistic Regression Weights")
    # label_panel(ax_r, "r")

    # ------------------------------------------------------------------
    # Row 5: (s/t) Cmp.17 sender/receiver PaCMAP, (u/v) GZMA-F2R heatmaps,
    #        (w/x) Cmp.17 gene weights (NK sender, CD8 T receiver)
    # ------------------------------------------------------------------
    row5 = subfigs[4].subplots(1, 6, width_ratios=[1, 1, 1, 1, 1.6, 1.6])
    ax_s, ax_t, ax_u, ax_v, ax_w, ax_x = row5

    plot_wc_pacmap(X, CMP_NK, ax_s, cbarMax=0.3, factor_matrix="B")
    ax_s.set_title(f"Cmp. {CMP_NK}: Sender Cells")
    label_panel(ax_s, "s")

    plot_wc_pacmap(X, CMP_NK, ax_t, cbarMax=0.3, factor_matrix="C")
    ax_t.set_title(f"Cmp. {CMP_NK}: Receiver Cells")
    label_panel(ax_t, "t")

    nk_sender = _ordered_population(X, celltype_col, "NK cells", CMP_NK, "sender")
    nk_receiver = _ordered_population(X, celltype_col, "NK cells", CMP_NK, "receiver")
    df = binned_expression_product_matrix(nk_sender, nk_receiver, "GZMA", "F2R")
    sns.heatmap(df, ax=ax_u, cmap="rocket")
    ax_u.set_title("GZMA-F2R Interaction")
    ax_u.set_xlabel("Receiver NK Population")
    ax_u.set_ylabel("Sender NK Population")
    ax_u.set_xticks([])
    ax_u.set_yticks([])
    label_panel(ax_u, "u")

    cd8t_sender = _ordered_population(X, celltype_col, "CD8 T cells", CMP_NK, "sender")
    cd8t_receiver = _ordered_population(
        X, celltype_col, "CD8 T cells", CMP_NK, "receiver"
    )
    df = binned_expression_product_matrix(cd8t_sender, cd8t_receiver, "GZMA", "F2R")
    sns.heatmap(df, ax=ax_v, cmap="rocket")
    ax_v.set_title("GZMA-F2R Interaction")
    ax_v.set_xlabel("Receiver CD8 T Population")
    ax_v.set_ylabel("Sender CD8 T Population")
    ax_v.set_xticks([])
    ax_v.set_yticks([])
    label_panel(ax_v, "v")

    _gene_weight_placeholder(ax_w, "w", "Top 10% Vs. Rest of Sender NK cells")
    # from ..logreg import population_gene_logreg_weights
    # nk_weights = population_gene_logreg_weights(
    #     X[X.obs[celltype_col] == "NK cells"], CMP_NK, "sender", pos=True
    # )
    # _gene_weight_barplot(nk_weights, ax_w)
    # ax_w.set_title("Top 10% Vs. Rest of Sender NK cells")
    # ax_w.set_xlabel(f"Cmp. {CMP_NK} Logistic Regression Weights")
    # label_panel(ax_w, "w")

    _gene_weight_placeholder(ax_x, "x", "Top 10% Vs. Rest of Receiver CD8 T cells")
    # cd8t_weights = population_gene_logreg_weights(
    #     X[X.obs[celltype_col] == "CD8 T cells"], CMP_NK, "receiver", pos=True
    # )
    # _gene_weight_barplot(cd8t_weights, ax_x)
    # ax_x.set_title("Top 10% Vs. Rest of Receiver CD8 T cells")
    # ax_x.set_xlabel(f"Cmp. {CMP_NK} Logistic Regression Weights")
    # label_panel(ax_x, "x")

    return f
