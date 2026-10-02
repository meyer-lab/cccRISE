"""
Figure 2: cccRISE resolves COVID-19-associated communication patterns to
specific immune subpopulations.

Panel (a) is a hand-drawn schematic with no corresponding plotting script
and is left as a placeholder.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from ...anova import get_sample_to_group
from ...tensor import run_pseudobulk_workflow
from ..common import get_covid_data
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_interaction_heatmap,
    plot_lr_factors,
)
from .commonFuncs.plotGeneral import rotate_yaxis
from .commonFuncs.plotPaCMAP import plot_labels_pacmap, plot_wc_pacmap

CCC_RISE_CMP = 6
PSEUDOBULK_CMP = 8


def label_panel(ax, letter):
    ax.text(
        -0.15,
        1.15,
        letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=12,
        va="top",
    )


def makeFigure():
    X = get_covid_data(project_cells=True, do_embedding=True)
    X_pseudo, _ = run_pseudobulk_workflow(X.copy(), cp_rank=PSEUDOBULK_CMP)

    condition_column = "sample"
    group_col = "condition"
    sample_to_group = get_sample_to_group(X, condition_column, group_col)

    f = plt.figure(figsize=(22, 16), layout="constrained")
    outer = gridspec.GridSpec(3, 1, figure=f, height_ratios=[1.1, 1.1, 1.0])

    # ------------------------------------------------------------------
    # Row 1: (a) schematic placeholder + (b) factor matrices
    # ------------------------------------------------------------------
    row1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], width_ratios=[1.3, 3]
    )

    ax_a = f.add_subplot(row1[0, 0])
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

    row1b = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=row1[0, 1], width_ratios=[1, 1, 1, 1.6]
    )
    ax_b = [f.add_subplot(row1b[0, i]) for i in range(4)]

    plot_condition_factors(
        X,
        ax_b[0],
        cond=condition_column,
        cond_group_labels=sample_to_group,
        group_cond=True,
        normalize=True,
    )
    ax_b[0].set_title("Condition Factor")

    plot_eigenstate_factors(X, ax_b[1], factor_type="B")
    ax_b[1].set_title("Sender Eigen-State Factor")
    rotate_yaxis(ax_b[1], rotation=0)

    plot_eigenstate_factors(X, ax_b[2], factor_type="C")
    ax_b[2].set_title("Receiver Eigen-State Factor")
    rotate_yaxis(ax_b[2], rotation=0)

    plot_lr_factors(X, ax_b[3], trim=True, weight=0.06)
    ax_b[3].set_title("Ligand-Receptor Pair")

    label_panel(ax_b[0], "b")

    # ------------------------------------------------------------------
    # Row 2: (c) condition PaCMAP, (d) cell type PaCMAP,
    #        (e) condition factor correlation, (f) LR factor correlation,
    #        (g) condition component scatter, (h) LR component scatter
    # ------------------------------------------------------------------
    row2 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[1])
    ax_c, ax_d, ax_e, ax_f, ax_g, ax_h = (f.add_subplot(row2[0, i]) for i in range(6))

    pal_cond = sns.color_palette("Set2", n_colors=3).as_hex()
    plot_labels_pacmap(X, labelType="condition", ax=ax_c, color_key=pal_cond)
    ax_c.set_title("Condition")
    label_panel(ax_c, "c")

    pal_celltype = sns.color_palette("Set3").as_hex()
    plot_labels_pacmap(X, labelType="celltype", ax=ax_d, color_key=pal_celltype)
    ax_d.set_title("Cell Type")
    label_panel(ax_d, "d")

    A_factor = X.uns["A"]
    A_factor_pseudo = X_pseudo.uns["A"]
    lr_factor = X.uns["D"]
    lr_factor_pseudo = X_pseudo.uns["D"]
    cmp = A_factor.shape[1]

    r_mat = np.zeros((cmp, cmp))
    for i in range(cmp):
        for j in range(cmp):
            r_mat[i, j] = pearsonr(A_factor[:, i], A_factor_pseudo[:, j])[0]
    r_df = pd.DataFrame(
        r_mat,
        index=pd.Index([f"{i + 1}" for i in range(cmp)]),
        columns=pd.Index([f"{i + 1}" for i in range(cmp)]),
    )
    sns.heatmap(r_df, cmap="coolwarm", ax=ax_e, vmin=-1, vmax=1)
    ax_e.set_title("Condition Factor Correlation")
    rotate_yaxis(ax_e, rotation=0)
    ax_e.set_xlabel("Pseudobulk CPD Condition Component")
    ax_e.set_ylabel("cccRISE Condition Component")
    label_panel(ax_e, "e")

    _, idx_sc, idx_pseudo = np.intersect1d(
        X.uns["lr_pairs"], X_pseudo.uns["lr_pairs"], return_indices=True
    )
    lr_sc_aligned = lr_factor[idx_sc, :]
    lr_pseudo_aligned = lr_factor_pseudo[idx_pseudo, :]

    r_mat = np.zeros((cmp, cmp))
    for i in range(cmp):
        for j in range(cmp):
            r_mat[i, j] = pearsonr(lr_sc_aligned[:, i], lr_pseudo_aligned[:, j])[0]
    r_df = pd.DataFrame(
        r_mat,
        index=pd.Index([f"{i + 1}" for i in range(cmp)]),
        columns=pd.Index([f"{i + 1}" for i in range(cmp)]),
    )
    sns.heatmap(r_df, cmap="coolwarm", ax=ax_f, vmin=-1, vmax=1)
    ax_f.set_title("LR Factor Correlation")
    rotate_yaxis(ax_f, rotation=0)
    ax_f.set_xlabel("Pseudobulk CPD LR Component")
    ax_f.set_ylabel("cccRISE LR Component")
    label_panel(ax_f, "f")

    pal = sns.color_palette(palette="Set2", n_colors=len(sample_to_group.unique()))
    pal = pal.as_hex()
    color_map = dict(zip(sample_to_group.unique(), pal))
    colors = [
        color_map[sample_to_group.loc[condition]] for condition in sample_to_group.index
    ]

    ax_g.scatter(
        A_factor[:, CCC_RISE_CMP - 1], A_factor_pseudo[:, PSEUDOBULK_CMP - 1], c=colors
    )
    ax_g.set_xscale("log")
    ax_g.set_yscale("log")
    r = np.corrcoef(
        A_factor[:, CCC_RISE_CMP - 1], A_factor_pseudo[:, PSEUDOBULK_CMP - 1]
    )[0, 1]
    ax_g.set_xlabel(f"cccRISE Condition Component {CCC_RISE_CMP}")
    ax_g.set_ylabel(f"Pseudobulk CPD Condition Component {PSEUDOBULK_CMP}")
    ax_g.set_title(f"r = {r:.3f}")
    label_panel(ax_g, "g")

    sns.scatterplot(
        x=lr_sc_aligned[:, CCC_RISE_CMP - 1],
        y=lr_pseudo_aligned[:, PSEUDOBULK_CMP - 1],
        ax=ax_h,
        color="black",
    )
    r = np.corrcoef(
        lr_sc_aligned[:, CCC_RISE_CMP - 1], lr_pseudo_aligned[:, PSEUDOBULK_CMP - 1]
    )[0, 1]
    ax_h.set_xlabel(f"cccRISE LR Component {CCC_RISE_CMP}")
    ax_h.set_ylabel(f"Pseudobulk CPD LR Component {PSEUDOBULK_CMP}")
    ax_h.set_xscale("log")
    ax_h.set_yscale("log")
    max_val = max(ax_h.get_xlim()[1], ax_h.get_ylim()[1])
    min_val = min(ax_h.get_xlim()[0], ax_h.get_ylim()[0])
    ax_h.set_xlim(min_val, max_val)
    ax_h.set_ylim(min_val, max_val)
    ticks = np.logspace(np.floor(np.log10(min_val)), np.ceil(np.log10(max_val)), num=5)
    ax_h.set_xticks(ticks)
    ax_h.set_yticks(ticks)
    ax_h.set_title(f"r = {r:.3f}")
    label_panel(ax_h, "h")

    # ------------------------------------------------------------------
    # Row 3: (i) sender PaCMAP assoc., (j) mDC weight violin,
    #        (k) receiver PaCMAP assoc., (l/m) CCL19-CCR7 heatmaps
    # ------------------------------------------------------------------
    row3 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[2])
    ax_i, ax_j, ax_k, ax_l, ax_m = (f.add_subplot(row3[0, i]) for i in range(5))

    plot_wc_pacmap(X, CCC_RISE_CMP, ax_i, cbarMax=0.3, factor_matrix="B")
    ax_i.set_title(f"Cmp. {CCC_RISE_CMP} Sender Cells")
    label_panel(ax_i, "i")

    X_mdc = X[X.obs["celltype"] == "mDC"]
    X_mdc_weights = X_mdc.obsm["sc_B"][:, CCC_RISE_CMP - 1]
    sns.violinplot(data=X_mdc_weights, ax=ax_j)
    ax_j.set_ylim(-0.1, 0.85)
    ax_j.set_xlabel("mDC Weight Distribution")
    ax_j.set_ylabel(f"Sender Cell Component {CCC_RISE_CMP} Association")
    label_panel(ax_j, "j")

    plot_wc_pacmap(X, CCC_RISE_CMP, ax_k, cbarMax=0.3, factor_matrix="C")
    ax_k.set_title(f"Cmp. {CCC_RISE_CMP} Receiver Cells")
    label_panel(ax_k, "k")

    plot_interaction_heatmap(
        X,
        cmp=CCC_RISE_CMP,
        ligand="CCL19",
        receptor="CCR7",
        ax=ax_l,
        sender_celltype="mDC",
        receiver_celltype="mDC",
        vmax=0.12,
        sender_label="Sender mDCs",
        receiver_label="Receiver mDCs",
    )
    label_panel(ax_l, "l")

    plot_interaction_heatmap(
        X,
        cmp=CCC_RISE_CMP,
        ligand="CCL19",
        receptor="CCR7",
        ax=ax_m,
        sender_celltype="mDC",
        receiver_celltype="B",
        vmax=0.12,
        sender_label="Sender mDCs",
        receiver_label="Receiver B Cells",
    )
    label_panel(ax_m, "m")

    return f
