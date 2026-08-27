"""Generate figures for the CCC-RISE tutorial documentation."""

from typing import cast

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Create output directory
output_dir = Path(__file__).parent / "_static" / "tutorial_images"
output_dir.mkdir(parents=True, exist_ok=True)

# Import CCC-RISE modules
from cellcommunicationpf2.figures.commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_lr_factors,
    plot_lr_factors_partial,
)
from cellcommunicationpf2.figures.commonFuncs.plotGeneral import plot_fms_r2x_stability
from cellcommunicationpf2.figures.commonFuncs.plotPaCMAP import (
    plot_labels_pacmap,
    plot_wc_pacmap,
)
from cellcommunicationpf2.import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from cellcommunicationpf2.tensor import (
    calculate_fms_rise,
    calculate_interaction_tensor,
    rise_store_r2x,
    run_fms_r2x_analysis,
)
from cellcommunicationpf2.utils import (
    binned_expression_product_matrix,
    resample,
)

print("Loading dataset...")
X = import_balf_covid(gene_threshold=0.001, normalize=True)
lr_pairs = import_ligand_receptor_pairs()

# Add numerical indices for each sample (condition)
condition_column = "sample"
X = add_cond_idxs(X, condition_column)

print("\n=== RISE Rank Selection ===")
print("Running RISE FMS/R2X analysis...")
rise_ranks = list(range(5, 41, 5))
runs = 3

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

print(f"Testing RISE ranks: {rise_ranks[0]}-{rise_ranks[-1]}, {runs} runs...")
rows = []
for i in rise_ranks:
    dataX, r2x = rise_store_r2x(X, rank=i)
    for j in range(runs):
        sampledX, _ = rise_store_r2x(
            resample(X, condition_name=condition_column),
            rank=i,
        )
        fms_score = calculate_fms_rise(dataX, sampledX)
        rows.append({"Run": j, "Component": i, "FMS": fms_score, "R2X": r2x})

df_rise = pd.DataFrame(rows)
plot_fms_r2x_stability(df_rise, ax, title_prefix="RISE", x_label="RISE Rank")

plt.tight_layout()
plt.savefig(output_dir / "step1_rise_fms_r2x.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n=== CPD Rank Selection ===")
print("Calculating interaction tensor for CPD rank selection...")
rise_rank_for_cpd = 35

# Process with RISE and calculate interaction tensor
print(f"Fitting RISE with rank={rise_rank_for_cpd} and building interaction tensor...")
X, _ = rise_store_r2x(X, rank=rise_rank_for_cpd, n_iter_max=1000, tolerance=1e-9)
interaction_tensor = calculate_interaction_tensor(X, lr_pairs)
print(f"Interaction tensor shape: {interaction_tensor.shape}")

# Run actual CPD FMS/R2X analysis
rank_list = list(range(1, 11, 2))
runs = 3
print(f"Testing CPD ranks: {rank_list}, {runs} runs...")
df = run_fms_r2x_analysis(
    interaction_tensor, rank_list=rank_list, runs=runs, svd_init="random"
)

# Figure 2: CPD FMS and R2X
print("Generating Figure 2: CPD rank selection...")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.lineplot(data=df, x="Component", y="FMS", ax=ax[0], label="FMS")
ax[0].set_ylim(0, 1)
ax[0].set_xlabel("CPD Rank")

sns.lineplot(data=df, x="Component", y="R2X", ax=ax[1], color="orange", label="R2X")
ax[1].set_ylim(0, np.max(df["R2X"]) + 0.02)
ax[1].set_xlabel("CPD Rank")

plt.tight_layout()
plt.savefig(output_dir / "step2_cpd_fms_r2x.png", dpi=150, bbox_inches="tight")
plt.close()

from cellcommunicationpf2.figures.common import get_covid_data

print("\n=== Running CCC-RISE Factorization ===")

X = get_covid_data(project_cells=True, do_embedding=True)

# Figure 3: Individual Factor Visualizations
print("Generating Figure 3: Individual factor visualizations...")

# Prepare condition grouping
group_col = "condition"
sample_to_group = (
    cast(pd.DataFrame, X.obs)
    .drop_duplicates(subset=[condition_column, group_col])
    .set_index(condition_column)[group_col]
)

# Factor A: Condition factors
fig, ax = plt.subplots(figsize=(6, 8))
plot_condition_factors(
    X,
    ax,
    cond=condition_column,
    cond_group_labels=sample_to_group,
    group_cond=True,
    normalize=True,
)
ax.set_title("Factor A: Condition")
# Take off legend
legend = ax.get_legend()
if legend is not None:
    legend.remove()
plt.savefig(output_dir / "step3_factor_a.png", dpi=150, bbox_inches="tight")
plt.close()

# Factor B: Sender cell eigenstates
fig, ax = plt.subplots(figsize=(6, 8))
plot_eigenstate_factors(X, ax, factor_type="B")
ax.set_title("Factor B: Sender Cell Eigenstate")
plt.tight_layout()
plt.savefig(output_dir / "step3_factor_b.png", dpi=150, bbox_inches="tight")
plt.close()

# Factor C: Receiver cell eigenstates
fig, ax = plt.subplots(figsize=(6, 8))
plot_eigenstate_factors(X, ax, factor_type="C")
ax.set_title("Factor C: Receiver Cell Eigenstate")
plt.tight_layout()
plt.savefig(output_dir / "step3_factor_c.png", dpi=150, bbox_inches="tight")
plt.close()

# Factor D: Ligand-receptor pairs
fig, ax = plt.subplots(figsize=(6, 8))
plot_lr_factors(X, ax, trim=True, weight=0.08)
ax.set_title("Factor D: LR Pairs")
plt.tight_layout()
plt.savefig(output_dir / "step3_factor_d.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 5: PaCMAP Embeddings with Three Subplots
print("Generating Figure 5: PaCMAP embeddings...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Cell type
if "celltype" in X.obs.columns:
    pal = sns.color_palette(palette="Set3")
    pal = pal.as_hex()
    plot_labels_pacmap(X, labelType="celltype", ax=axes[0], color_key=pal)
    axes[0].set_title("Cells Colored by Cell Type")

# Condition
if group_col in X.obs.columns:
    pal = sns.color_palette("Set2")
    pal = [pal[0], pal[1], pal[2]]
    pal = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b in pal
    ]
    plot_labels_pacmap(X, labelType=group_col, ax=axes[1], color_key=pal)
    axes[1].set_title("Cells Colored by Condition")

# Sample
plot_labels_pacmap(X, labelType=condition_column, ax=axes[2])
axes[2].set_title("Cells Colored by Sample")

plt.tight_layout()
plt.savefig(output_dir / "step5_pacmap_embedding.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 7: Component Loadings for Component 6
print("Generating Figure 7: Component loadings for component 6...")
component = 6  # Component to visualize
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Sender cell weightings
plot_wc_pacmap(X, component - 1, ax[0], factor_matrix="B", cbarMax=0.3)
ax[0].set_title(f"Cmp.{component} Sender Cells")

# Receiver cell weightings
plot_wc_pacmap(X, component - 1, ax[1], factor_matrix="C", cbarMax=0.3)
ax[1].set_title(f"Cmp.{component} Receiver Cells")

plt.tight_layout()
plt.savefig(
    output_dir / "step7_all_component_weightings.png", dpi=150, bbox_inches="tight"
)
plt.close()

# Figure 9: Violin Plot of Weights for Component 6
print("Generating Figure 9: Violin plot of component weights...")
component = 6
if "celltype" in X.obs.columns:
    # Use mDC cell type if available
    cell_type = (
        "mDC"
        if "mDC" in X.obs["celltype"].cat.categories
        else X.obs["celltype"].cat.categories[0]
    )
    X_celltype = X[X.obs["celltype"] == cell_type]
    sender_weights = X_celltype.obsm["sc_B"][:, component - 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.violinplot(data=sender_weights, ax=ax)
    ax.set_ylim(-0.1, max(sender_weights) + 0.1)
    ax.set_xlabel(f"{cell_type} Weight Distribution")
    ax.set_ylabel(f"Sender Cell Component {component} Association")
    ax.set_title(f"Distribution of Component {component} Weights in {cell_type}")
    plt.tight_layout()
    plt.savefig(output_dir / "step9_violin_weights.png", dpi=150, bbox_inches="tight")
    plt.close()

# Figure 7: Top and Bottom LR Pairs for Component 6
print("Generating Figure 7: Top and bottom LR pairs for component 6...")
cmp = 6
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

plot_lr_factors_partial(X, cmp, ax[0], geneAmount=10, top=True)
ax[0].set_title(f"Component {cmp}: Top 10 LR Pairs")

plot_lr_factors_partial(X, cmp, ax[1], geneAmount=10, top=False)
ax[1].set_title(f"Component {cmp}: Bottom 10 LR Pairs")

plt.tight_layout()
plt.savefig(output_dir / "step7_top_lr_pairs.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 10: Expression Product Heatmap for Component 6
print("Generating Figure 10: Expression product heatmap...")
ccc_rise_cmp = 6  # Component number (1-indexed)
ligand = "CCL19"
receptor = "CCR7"

if (
    ligand in X.var_names
    and receptor in X.var_names
    and "celltype" in X.obs.columns
    and "mDC" in X.obs["celltype"].cat.categories
):
    # Filter and sort sender cells by component weight
    X_mdc_sender = X[X.obs["celltype"] == "mDC"]
    X_mdc_sender = X_mdc_sender[
        np.argsort(-X_mdc_sender.obsm["sc_B"][:, ccc_rise_cmp - 1])
    ]

    # Filter and sort receiver cells by component weight
    X_mdc_receiver = X[X.obs["celltype"] == "mDC"]
    X_mdc_receiver = X_mdc_receiver[
        np.argsort(X_mdc_receiver.obsm["rc_C"][:, ccc_rise_cmp - 1])
    ]

    # Calculate expression product matrix for mDC -> mDC
    df = binned_expression_product_matrix(
        X_mdc_sender, X_mdc_receiver, ligand, receptor
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(df, ax=ax, cmap="viridis", vmax=0.12)
    ax.set_xlabel("Receiver mDCs")
    ax.set_ylabel("Sender mDCs")
    ax.set_title("CCL19-CCR7 Interaction")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(
        output_dir / "step10_expression_product.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
