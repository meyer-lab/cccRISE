# cccRISE - Cell-Cell Communication with RISE

cccRISE (Cell-Cell Communication with Reduction and Insight in Single-cell Exploration) is an unsupervised, tensor-based computational method designed for the integrative analysis of cell-cell communication in single-cell RNA sequencing (scRNA-seq) data across multiple experimental conditions. Built upon the CP (CANDECOMP/PARAFAC) tensor decomposition framework combined with PARAFAC2, cccRISE preserves the four-dimensional structure of multi-condition cell-cell communication data—conditions × sender cells × receiver cells × ligand-receptor pairs—instead of flattening it into conventional matrices.

cccRISE decomposes variation into distinct, interpretable patterns associated with experimental conditions, sender cells, receiver cells, and specific ligand-receptor interactions, providing a more nuanced and biologically meaningful analysis of intercellular communication dynamics. This approach enables the identification of coordinated signaling programs that change across conditions, offering insights into how cellular communication networks respond to different experimental perturbations.

cccRISE does not require prior cell-type labels or clustering, reducing bias and enabling discovery of novel communication patterns, while maintaining high resolution to identify condition-specific signaling subpopulations missed by pseudobulk or clustering-based approaches. Each resulting component is directly linked to specific conditions, sender cells, receiver cells, and ligand-receptor pairs, making the results biologically tractable.

- **Read the documentation** at [cccRISE Documentation](https://meyer-lab.github.io/cccRISE/).
- cccRISE uses the [AnnData](https://anndata.readthedocs.io/) format for handling single-cell data matrices.

## Installation

To add cccRISE to your Python environment, you can install it directly from GitHub:

```bash
pip install git+https://github.com/meyer-lab/cccRISE.git@main
```

Or add the following line to your `requirements.txt`:

```
git+https://github.com/meyer-lab/cccRISE.git@main
```

## Quick Start

cccRISE works with preprocessed AnnData objects containing single-cell RNA-seq data:

```python
from cccrise import run_ccc_rise_workflow
import pandas as pd

# Load your ligand-receptor pairs database
lr_pairs = pd.read_csv(
    "lr_pairs.csv"
)  # Should contain ligand-receptor interaction information

# Perform cccRISE tensor decomposition
fit = run_ccc_rise_workflow(
    adata=adata,
    rise_rank=20,
    lr_pairs=lr_pairs,
    rise_n_iter_max=300,
    cp_n_iter_max=4000,
    rise_tol=1e-8,
    cp_tol=1e-12,
    condition_column="sample",
    doEmbedding=True,
    random_state=42,
)
adata, r2x = fit["adata"], fit["r2x_cp"]

# `fit` also holds the CP factors, the RISE R2X and CP diagnostics.
# Results are stored in the AnnData object:
# - adata.uns["weights"]: Component weights
# - adata.uns["A"]: Condition factors
# - adata.uns["B"]: Sender cell factors
# - adata.uns["C"]: Receiver cell factors
# - adata.uns["D"]: Ligand-receptor pair factors
# - adata.uns["lr_pairs"]: Filtered ligand-receptor pairs
# - adata.uns["r2x"]: Variance explained by the decomposition
# - adata.obsm["projections"]: Cell projections
# - adata.obsm["sc_B"]: Sender cell embeddings
# - adata.obsm["rc_C"]: Receiver cell embeddings
# - adata.obsm["PaCMAP"]: PaCMAP embeddings (if doEmbedding=True)
```

See the [tutorial](https://meyer-lab.github.io/cccRISE/tutorial.html) for a complete workflow including preprocessing, rank selection, visualization, and interpretation.

## Key Features

- **4D Tensor decomposition**: Preserves the full structure of cell-cell communication data (conditions × sender cells × receiver cells × ligand-receptor pairs)
- **Unsupervised analysis**: No prior cell-type labels or clustering required
- **Integrated communication analysis**: Simultaneously analyzes sender cells, receiver cells, and specific ligand-receptor interactions
- **High resolution**: Identifies communication patterns and condition-specific signaling subpopulations
- **Interpretable results**: Components directly linked to conditions, sender/receiver cells, and ligand-receptor pairs
- **Condition-aware**: Captures how communication networks change across experimental conditions
- **Integrated workflow**: Built-in preprocessing, visualization, and interpretation tools

## Citation

If you use cccRISE in your work, please cite the cccRISE publication as follows:

**CCC-RISE resolves cellular communication across conditions at single-cell resolution**

Samuel Amidon\*, Ethan Hurt\*, Andrew Ramirez, Nathaniel Thomas, Daniel R. Calabrese, John R. Greenland, Aaron S. Meyer (\*equal contribution)

*Under revision at Molecular Systems Biology*, 2026.