API Reference
=============

Core Functions
--------------

Preprocessing
~~~~~~~~~~~~~

.. py:function:: prepare_dataset(X: anndata.AnnData, condition_name: str, geneThreshold: float, normalize: bool = False)
   :module: cellcommunicationpf2.import_data

   Preprocess single-cell RNA-seq data for CCC-RISE analysis.
   
   This function performs essential preprocessing steps including cell and gene filtering, 
   normalization, log transformation, and creation of condition indices required for 
   CCC-RISE decomposition.
   
   :param X: AnnData object containing raw count data in sparse matrix format.
   :type X: anndata.AnnData
   :param condition_name: Name of the column in X.obs that specifies experimental conditions for each cell.
   :type condition_name: str
   :param geneThreshold: Minimum mean expression threshold for gene filtering. Genes with mean expression below this value are removed.
   :type geneThreshold: float
   :param normalize: If True, performs normalization and log transformation. If False, keeps raw counts.
   :type normalize: bool, optional
   :returns: Preprocessed AnnData object with filtered cells/genes, added condition_unique_idxs column, and gene means in X.var["means"].
   :rtype: anndata.AnnData

.. py:function:: import_ligand_receptor_pairs(filename: str | Path = "cellcommunicationpf2/data/Human-2020-Jin-LR-pairs.csv.zst", update_interaction_names: bool = True)
   :module: cellcommunicationpf2.import_data

   Import ligand-receptor pairs from a compressed CSV file with caching.
   
   Loads a curated database of ligand-receptor pairs from CellChat. The function uses 
   LRU caching to avoid repeated file reads. For protein complexes, subunits are separated 
   by & (e.g., CD74&CD44).
   
   :param filename: Path to the compressed CSV file containing ligand-receptor pairs.
   :type filename: str, optional
   :param update_interaction_names: If True, updates interaction names to use standardized formatting (uppercase, & for complexes).
   :type update_interaction_names: bool, optional
   :returns: DataFrame with ligand and receptor columns containing gene names.
   :rtype: pd.DataFrame

.. py:function:: add_cond_idxs(X: anndata.AnnData, condition_key: str)
   :module: cellcommunicationpf2.import_data

   Add unique 0-indexed condition identifiers to an AnnData object.
   
   Creates a new column condition_unique_idxs in X.obs that maps condition labels 
   to 0-based integer indices. This is required for proper data partitioning in the 
   PARAFAC2 decomposition.
   
   :param X: AnnData object to add condition indices to.
   :type X: anndata.AnnData
   :param condition_key: Column name in X.obs containing condition identifiers.
   :type condition_key: str
   :returns: AnnData object with added condition_unique_idxs column in X.obs.
   :rtype: anndata.AnnData

Factor Storage & Reconstruction (xarray / NetCDF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: factors_to_xarray(cpd_factors: list[np.ndarray], cpd_weights: np.ndarray, filtered_lr_pairs: pd.DataFrame | list[str] | np.ndarray, condition_names: list[str] | np.ndarray, gene_names: list[str] | np.ndarray | None = None, pf2_factors: list[np.ndarray] | None = None, pf2_weights: np.ndarray | None = None, r2x_cp: float | None = None, r2x_pf2: float | None = None, attrs: dict | None = None)
   :module: cellcommunicationpf2.factors

   Convert PARAFAC2 and CP decomposition factor matrices into a compact, coordinate-annotated xarray Dataset.

   :param cpd_factors: List of 4 CP factor matrices: [A (condition), B (sender), C (receiver), D (LR)].
   :param cpd_weights: Component scaling weights for CP decomposition.
   :param filtered_lr_pairs: Ligand-receptor pair symbols matching dimension 3.
   :param condition_names: Labels for conditions / samples matching dimension 0.
   :param gene_names: List of gene identifiers matching single-cell features.
   :param pf2_factors: Optional list of PARAFAC2 factor matrices [A_pf2, B_pf2, C_pf2].
   :param pf2_weights: Optional PARAFAC2 component weights.
   :param r2x_cp: Variance explained by CP decomposition.
   :param r2x_pf2: Variance explained by PARAFAC2 decomposition.
   :param attrs: Additional metadata dictionary.
   :returns: Annotated xarray Dataset holding the factor matrices and coordinates.
   :rtype: xr.Dataset

.. py:function:: save_factors_netcdf(ds: xr.Dataset, filepath: str | Path)
   :module: cellcommunicationpf2.factors

   Save an xarray Dataset of factors to a compact NetCDF file (< 100 KB - 2 MB).

.. py:function:: load_factors_netcdf(filepath: str | Path)
   :module: cellcommunicationpf2.factors

   Load an xarray Dataset of factors from a NetCDF file.

.. py:function:: load_factors_to_anndata(factors_ds: xr.Dataset | str | Path, adata: anndata.AnnData | None = None, condition_column: str | None = None, do_embedding: bool = False, random_state: int | None = 0)
   :module: cellcommunicationpf2.factors

   Populate or construct an AnnData object with decomposition factors from an xarray Dataset.
   
   If ``adata`` is provided, computes single-cell factor projections (``obsm["projections"]``),
   sender loadings (``obsm["sc_B"]``), receiver loadings (``obsm["rc_C"]``), and optional 2D PaCMAP embeddings.
   If ``adata`` is None, creates a lightweight factor container AnnData in milliseconds without loading scRNA-seq matrices.

   :param factors_ds: Factor dataset or path to NetCDF file.
   :param adata: Optional single-cell AnnData to project onto.
   :param condition_column: Condition column name in adata.obs.
   :param do_embedding: If True, computes 2D PaCMAP on cell projections.
   :param random_state: Random seed for reproducibility.
   :returns: AnnData object populated with factors and projections.
   :rtype: anndata.AnnData


Factorization
~~~~~~~~~~~~~

.. py:function:: run_ccc_rise_workflow(adata: anndata.AnnData, rise_rank: int, lr_pairs: pd.DataFrame, cp_rank: int = None, condition_column: str = "sample", n_iter_max: int = 100, tol: float = 1e-3, random_state: int = None, complex_sep: str = None, doEmbedding: bool = True, svd_init: str = "svd")
   :module: cellcommunicationpf2.tensor

   Execute the complete CCC-RISE workflow including RISE decomposition, CPD factorization, and result storage.
   
   This is the main function for performing CCC-RISE analysis. It executes both the RISE 
   (PARAFAC2) decomposition of expression data and the CPD decomposition of the resulting 
   interaction tensor. The function decomposes cell-cell communication into four interpretable 
   factor matrices representing conditions, sender cells, receiver cells, and ligand-receptor pairs.
   
   :param adata: AnnData object with preprocessed scRNA-seq data. Must have condition_unique_idxs in adata.obs.
   :type adata: anndata.AnnData
   :param rise_rank: Number of PARAFAC2 components to extract from expression data. Typically chosen based on FMS and R²X analysis.
   :type rise_rank: int
   :param lr_pairs: DataFrame of ligand-receptor pairs with 'ligand' and 'receptor' columns.
   :type lr_pairs: pd.DataFrame
   :param cp_rank: Number of CPD components for factorizing the interaction tensor. If None, defaults to rise_rank.
   :type cp_rank: int, optional
   :param condition_column: Column name in adata.obs containing condition identifiers.
   :type condition_column: str, optional
   :param n_iter_max: Maximum iterations for decomposition.
   :type n_iter_max: int, optional
   :param tol: Convergence tolerance for optimization.
   :type tol: float, optional
   :param random_state: Random seed for reproducibility of decomposition.
   :type random_state: int, optional
   :param complex_sep: Separator for protein complexes in L-R pairs (typically "&").
   :type complex_sep: str, optional
   :param doEmbedding: If True, automatically computes PaCMAP embeddings for visualization and stores in adata.obsm["PaCMAP"].
   :type doEmbedding: bool, optional
   :param svd_init: Initialization method for CPD ('svd' or 'random').
   :type svd_init: str, optional
   :returns: Tuple containing the updated AnnData object with stored results and the R²X value (variance explained).
   :rtype: tuple[anndata.AnnData, float]

.. py:function:: calculate_interaction_tensor(adata: anndata.AnnData, lr_pairs: pd.DataFrame, complex_sep: str = "&")
   :module: cellcommunicationpf2.tensor

   Calculate the interaction tensor from an AnnData object that has already been processed with RISE.
   
   This function computes cell-cell communication scores for all ligand-receptor pairs across
   sender-receiver latent cell state pairs using the pre-computed RISE projections in `adata.obsm["projections"]`.
   The resulting tensor has dimensions (n_conditions × rise_rank × rise_rank × n_lr_pairs).
   
   :param adata: AnnData object containing expression data and RISE projections in ``obsm["projections"]``.
   :type adata: anndata.AnnData
   :param lr_pairs: DataFrame of ligand-receptor pairs with 'ligand' and 'receptor' columns.
   :type lr_pairs: pd.DataFrame
   :param complex_sep: Symbol that separates protein subunits in multimeric complexes (default: "&").
   :type complex_sep: str, optional
   :returns: Interaction tensor with dimensions (n_conditions × rise_rank × rise_rank × n_lr_pairs).
   :rtype: np.ndarray

.. py:function:: run_fms_r2x_analysis(interaction_tensor: np.ndarray, rank_list: list[int] = None, runs: int = 1, svd_init: str = "svd")
   :module: cellcommunicationpf2.tensor

   Run Factor Match Score (FMS) and R²X analysis across different CPD ranks to assess stability and fit.
   
   This function helps determine the optimal CPD rank by evaluating model stability (FMS) 
   through bootstrap resampling and variance explained (R²X). FMS values above 0.6 indicate 
   reliable, reproducible components. Use this before finalizing your CPD rank choice.
   
   :param interaction_tensor: Pre-computed interaction tensor from calculate_interaction_tensor().
   :type interaction_tensor: np.ndarray
   :param rank_list: List of CPD ranks to test (e.g., [1, 3, 5, 7, 9]). If None, defaults to [1, 3].
   :type rank_list: list of int, optional
   :param runs: Number of bootstrap runs for stability assessment.
   :type runs: int, optional
   :param svd_init: Initialization method ('svd' or 'random').
   :type svd_init: str, optional
   :returns: DataFrame with columns ['Run', 'Component', 'FMS', 'R2X'] for each rank and run.
   :rtype: pd.DataFrame


Visualization Functions
-----------------------

Rank Selection
~~~~~~~~~~~~~~

.. py:function:: plot_fms_r2x_diff_ranks(X: anndata.AnnData, condition_name: str, ax1: matplotlib.axes.Axes, ax2: matplotlib.axes.Axes, ranksList: list[int], runs: int)
   :module: cellcommunicationpf2.figures.commonFuncs.plotGeneral

   Plot Factor Match Score (FMS) and R²X across different RISE ranks for rank selection.
   
   This function evaluates multiple RISE ranks by computing FMS (stability) and R²X 
   (variance explained) metrics. It performs bootstrap resampling to assess component 
   reproducibility. Use this to determine the optimal RISE rank before running the full 
   CCC-RISE workflow.
   
   :param X: AnnData object with preprocessed data. Must have condition_unique_idxs in X.obs.
   :type X: anndata.AnnData
   :param condition_name: Column name in X.obs containing condition identifiers for bootstrap resampling.
   :type condition_name: str
   :param ax1: Matplotlib axes object for plotting FMS values.
   :type ax1: matplotlib.axes.Axes
   :param ax2: Matplotlib axes object for plotting R²X values.
   :type ax2: matplotlib.axes.Axes
   :param ranksList: List of RISE ranks to evaluate (e.g., [5, 10, 15, 20, 25, 30, 35, 40]).
   :type ranksList: list of int
   :param runs: Number of bootstrap runs for stability assessment per rank.
   :type runs: int


Factor Plotting
~~~~~~~~~~~~~~~

.. py:function:: plot_condition_factors(data: anndata.AnnData, ax: matplotlib.axes.Axes, cond: str = "Condition", cond_group_labels: pd.Series = None, color_key: list = None, group_cond: bool = False, normalize: bool = False)
   :module: cellcommunicationpf2.figures.commonFuncs.plotFactors

   Plot Factor A (condition factors) as a heatmap showing how conditions contribute to components.
   
   This visualization shows how each experimental condition (rows) contributes to each CCC-RISE 
   component (columns). High values indicate strong association between a condition and a 
   component's communication pattern.

   :param data: AnnData object with stored CCC-RISE results. Must contain data.uns["A"] and data.obs[cond].
   :type data: anndata.AnnData
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param cond: Name of column in data.obs containing condition labels.
   :type cond: str, optional
   :param cond_group_labels: Series mapping conditions to group labels for colored row annotations.
   :type cond_group_labels: pd.Series, optional
   :param color_key: Custom colors for condition group labels.
   :type color_key: list, optional
   :param group_cond: If True and cond_group_labels provided, sorts conditions by group.
   :type group_cond: bool, optional
   :param normalize: If True, normalizes each component to [-1, 1] range.
   :type normalize: bool, optional

.. py:function:: plot_eigenstate_factors(data: anndata.AnnData, ax: matplotlib.axes.Axes, factor_type: str)
   :module: cellcommunicationpf2.figures.commonFuncs.plotFactors

   Plot Factor B (sender eigenstates) or Factor C (receiver eigenstates) as a heatmap.
   
   Eigenstate factors represent the underlying cell state patterns across components in the 
   latent RISE space. Each row represents a latent dimension from RISE and each column 
   represents a CCC-RISE component.
   
   :param data: AnnData object with stored CCC-RISE results. Must contain data.uns["B"] or data.uns["C"].
   :type data: anndata.AnnData
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param factor_type: Either "B" for sender eigenstates or "C" for receiver eigenstates.
   :type factor_type: str

.. py:function:: plot_lr_factors(data: anndata.AnnData, ax: matplotlib.axes.Axes, trim: bool = True, weight: float = 0.08)
   :module: cellcommunicationpf2.figures.commonFuncs.plotFactors

   Plot Factor D (ligand-receptor pairs) as a heatmap showing which L-R pairs drive each component.
   
   This visualization reveals coordinated signaling programs by showing which ligand-receptor 
   pairs (rows) are highly weighted in each component (columns). Only L-R pairs with maximum 
   absolute weight above the threshold are displayed.
   
   :param data: AnnData object with stored CCC-RISE results. Must contain data.uns["D"] and data.uns["lr_pairs"].
   :type data: anndata.AnnData
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param trim: If True, filters L-R pairs based on the weight parameter.
   :type trim: bool, optional
   :param weight: Minimum absolute weight threshold for including L-R pairs.
   :type weight: float, optional

.. py:function:: plot_lr_factors_partial(X: anndata.AnnData, cmp: int, ax: matplotlib.axes.Axes, geneAmount: int = 5, top: bool = True)
   :module: cellcommunicationpf2.figures.commonFuncs.plotFactors

   Plot the top or bottom weighted ligand-receptor pairs for a specific component as a bar plot.
   
   This visualization identifies the most positively or negatively weighted L-R pairs for a 
   single component, revealing which specific interactions are most associated with that 
   communication pattern.
   
   :param X: AnnData object with stored CCC-RISE results. Must contain X.uns["D"] and X.uns["lr_pairs"].
   :type X: anndata.AnnData
   :param cmp: Component number to visualize (1-indexed).
   :type cmp: int
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param geneAmount: Number of L-R pairs to display.
   :type geneAmount: int, optional
   :param top: If True, shows highest-weighted pairs; if False, shows lowest-weighted pairs.
   :type top: bool, optional


PaCMAP Visualization
~~~~~~~~~~~~~~~~~~~~

.. py:function:: plot_labels_pacmap(X: anndata.AnnData, labelType: str, ax: matplotlib.axes.Axes, condition: list = None, cmap: str = "tab20", color_key: list = None)
   :module: cellcommunicationpf2.figures.commonFuncs.plotPaCMAP

   Plot PaCMAP embedding colored by categorical labels (cell type or condition).
   
   This visualization shows the overall structure of the cell embedding in the latent 
   communication space, revealing how cells cluster by cell type or experimental condition 
   based on their communication patterns.
   
   :param X: AnnData object with RISE decomposition results. Must contain X.obsm["PaCMAP"] and X.obs[labelType].
   :type X: anndata.AnnData
   :param labelType: Name of column in X.obs containing categorical labels to color by.
   :type labelType: str
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param condition: If provided, only highlights cells from these specific conditions.
   :type condition: list of str, optional
   :param cmap: Matplotlib colormap name for coloring categories.
   :type cmap: str, optional
   :param color_key: Custom list of colors for categories.
   :type color_key: list, optional

.. py:function:: plot_wc_pacmap(X: anndata.AnnData, cmp: int, ax: matplotlib.axes.Axes, cbarMax: float = 1.0, factor_matrix: str = None)
   :module: cellcommunicationpf2.figures.commonFuncs.plotPaCMAP

   Plot PaCMAP embedding colored by weighted projections for a specific component.
   
   This visualization shows which cells contribute most strongly to a specific component by 
   coloring them according to their sender (Factor B) or receiver (Factor C) weights. Higher 
   values indicate stronger association with the component's communication pattern.
   
   :param X: AnnData object with RISE decomposition results. Must contain X.obsm["PaCMAP"] and X.obsm["sc_B"] or X.obsm["rc_C"].
   :type X: anndata.AnnData
   :param cmp: Component number to visualize (1-indexed).
   :type cmp: int
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param cbarMax: Maximum value for the color scale.
   :type cbarMax: float, optional
   :param factor_matrix: Either "B" for sender weights or "C" for receiver weights.
   :type factor_matrix: str


Analysis Functions
~~~~~~~~~~~~~~~~~~

.. py:function:: expression_product_matrix(X1: anndata.AnnData, X2: anndata.AnnData, ligand: str, receptor: str)
   :module: cellcommunicationpf2.utils

   Calculate the expression product matrix for a specific ligand-receptor pair between cell populations.
   
   For each cell in X1 (senders) and each cell in X2 (receivers), this computes the product 
   of ligand expression (in sender) and receptor expression (in receiver). This represents 
   the potential for that specific L-R interaction between each sender-receiver cell pair.
   
   :param X1: AnnData object containing sender cells.
   :type X1: anndata.AnnData
   :param X2: AnnData object containing receiver cells.
   :type X2: anndata.AnnData
   :param ligand: Ligand gene name (must be present in X1.var_names).
   :type ligand: str
   :param receptor: Receptor gene name (must be present in X2.var_names).
   :type receptor: str
   :returns: DataFrame with sender cells as rows and receiver cells as columns, values are expression products.
   :rtype: pd.DataFrame

.. py:function:: binned_expression_product_matrix(X1: anndata.AnnData, X2: anndata.AnnData, ligand: str, receptor: str, n_bins: int = 10)
   :module: cellcommunicationpf2.utils

   Compute the expression product matrix for a ligand-receptor pair, binned into an
   n_bins x n_bins matrix of block means.

   Equivalent to binning ``expression_product_matrix(X1, X2, ligand, receptor)`` into an
   n_bins x n_bins matrix of block means, but computed without ever materializing the full
   len(X1) x len(X2) outer product: since block entries are a product of independent per-cell
   terms, the mean of a block is just the product of the row- and column-block means of the
   two expression vectors.

   :param X1: AnnData object containing sender cells.
   :type X1: anndata.AnnData
   :param X2: AnnData object containing receiver cells.
   :type X2: anndata.AnnData
   :param ligand: Ligand gene name (must be present in X1.var_names).
   :type ligand: str
   :param receptor: Receptor gene name (must be present in X2.var_names).
   :type receptor: str
   :param n_bins: Number of bins per axis.
   :type n_bins: int
   :returns: n_bins x n_bins DataFrame with averaged expression products.
   :rtype: pd.DataFrame
