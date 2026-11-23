API Reference
=============

Core Functions
--------------

Preprocessing
~~~~~~~~~~~~~

.. py:function:: prepare_dataset(X: anndata.AnnData, condition_name: str, geneThreshold: float, normalize: bool = False)
   :module: cccrise.import_data

   Preprocess single-cell RNA-seq data for cccRISE analysis.
   
   This function performs essential preprocessing steps including cell and gene filtering, 
   normalization, log transformation, and creation of condition indices required for 
   cccRISE decomposition.
   
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

.. py:function:: import_ligand_receptor_pairs(filename: str | Path = "cccrise/data/Human-2020-Jin-LR-pairs.parquet", update_interaction_names: bool = True)
   :module: cccrise.import_data

   Import ligand-receptor pairs from a compressed parquet file with caching.
   
   Loads a curated database of ligand-receptor pairs from CellChat. The function uses 
   LRU caching to avoid repeated file reads. For protein complexes, subunits are separated 
   by & (e.g., CD74&CD44).
   
   :param filename: Path to the parquet file containing ligand-receptor pairs.
   :type filename: str, optional
   :param update_interaction_names: If True, updates interaction names to use standardized formatting (uppercase, & for complexes).
   :type update_interaction_names: bool, optional
   :returns: DataFrame with ligand and receptor columns containing gene names.
   :rtype: pd.DataFrame

.. py:function:: add_cond_idxs(X: anndata.AnnData, condition_key: str)
   :module: cccrise.import_data

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
   :module: cccrise.factors

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
   :module: cccrise.factors

   Save an xarray Dataset of factors to a compact NetCDF file (a few MB).

.. py:function:: save_ccc_rise_fit(path: str | Path, adata: anndata.AnnData, cp_fit: dict, condition_column: str, attrs: dict | None = None)
   :module: cccrise.factors

   Write a CCC-RISE fit (a RISE-fit AnnData and a ``fit_ccc_cp`` / ``run_ccc_rise_workflow`` result) as the factors file ``load_factors_to_anndata`` reads. Per-cell projections are not stored (``compute_cell_projections`` recomputes them); the file fingerprints the fit's cells and gene centering, and stores the PaCMAP embedding (float16) when present.

.. py:function:: compute_cell_projections(factors_ds: xr.Dataset, adata: anndata.AnnData)
   :module: cccrise.factors

   Recompute the per-cell RISE projections of ``adata`` (the fit's own data, checked against the file's fingerprints) under a stored RISE fit, without densifying ``adata.X``.

.. py:function:: add_fit_fingerprint(ds: xr.Dataset, adata: anndata.AnnData)
   :module: cccrise.factors

   Record in a factors Dataset the data its RISE fit was made on (a hash of the cell IDs and the per-gene centering), which ``compute_cell_projections`` requires and checks.

.. py:function:: load_factors_netcdf(filepath: str | Path)
   :module: cccrise.factors

   Load an xarray Dataset of factors from a NetCDF file.

.. py:function:: load_factors_to_anndata(factors_ds: xr.Dataset | str | Path, adata: anndata.AnnData | None = None, condition_column: str | None = None, do_embedding: bool = False, random_state: int | None = 42)
   :module: cccrise.factors

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

.. py:function:: run_ccc_rise_workflow(adata: anndata.AnnData, rise_rank: int, lr_pairs: pd.DataFrame, cp_rank: int = None, condition_column: str = "sample", random_state: int = None, complex_sep: str = "&", communication_score: str = "expression_product", hill_k: float = 0.5, backend: str = None, compress: bool = False, cp_device: str = None, cp_dtype: str = "float32", doEmbedding: bool = False, keep_tensor: bool = False, *, rise_n_iter_max: int, cp_n_iter_max: int, rise_tol: float, cp_tol: float)
   :module: cccrise.ccc_rise

   Execute the complete cccRISE workflow: RISE decomposition, interaction tensor, CPD factorization, and result storage.
   
   This is the main function for performing cccRISE analysis. It executes both the RISE 
   (PARAFAC2) decomposition of expression data and the CPD decomposition of the resulting 
   interaction tensor. The function decomposes cell-cell communication into four interpretable 
   factor matrices representing conditions, sender cells, receiver cells, and ligand-receptor pairs.
   
   :param adata: AnnData object with preprocessed scRNA-seq data.
   :type adata: anndata.AnnData
   :param rise_rank: Number of PARAFAC2 components to extract from expression data. Typically chosen based on FMS and R²X analysis.
   :type rise_rank: int
   :param lr_pairs: DataFrame of ligand-receptor pairs with 'ligand' and 'receptor' columns.
   :type lr_pairs: pd.DataFrame
   :param cp_rank: Number of CPD components for factorizing the interaction tensor. If None, defaults to rise_rank.
   :type cp_rank: int, optional
   :param condition_column: Column name in adata.obs containing condition identifiers.
   :type condition_column: str, optional
   :param random_state: Random seed for the RISE fit, the CPD fit and PaCMAP.
   :type random_state: int, optional
   :param complex_sep: Separator for protein complexes in L-R pairs.
   :type complex_sep: str, optional
   :param communication_score: Scoring strategy for communication (see ``calculate_interaction_tensor``).
   :type communication_score: str, optional
   :param hill_k: Half-maximal constant when using Hill kinetics scoring.
   :type hill_k: float, optional
   :param backend: RISE backend ('cpu', 'cupy', 'mlx', or None to auto-detect).
   :type backend: str, optional
   :param compress: Fit RISE in a compressed gene/cell subspace (see ``fit_rise``). The replicate runners compress; the core runner does not. CPD is never compressed.
   :type compress: bool, optional
   :param cp_device: Device the CPD fit runs on ('gpu', 'cpu', or None for the GPU when there is one).
   :type cp_device: str, optional
   :param cp_dtype: Precision of the CPD fit ('float32' or 'float64').
   :type cp_dtype: str, optional
   :param doEmbedding: If True, computes PaCMAP embeddings for visualization and stores in adata.obsm["PaCMAP"].
   :type doEmbedding: bool, optional
   :param keep_tensor: If True, the interaction tensor is returned.
   :type keep_tensor: bool, optional
   :param rise_n_iter_max: Maximum iterations for the RISE fit.
   :type rise_n_iter_max: int
   :param cp_n_iter_max: Maximum iterations for the CP fit.
   :type cp_n_iter_max: int
   :param rise_tol: Convergence tolerance for the RISE fit.
   :type rise_tol: float
   :param cp_tol: Convergence tolerance for the CP fit.
   :type cp_tol: float
   :returns: Dict with keys 'adata', 'tensor' (None unless keep_tensor), 'filtered_lr_pairs', 'cp_weights', 'cp_factors', 'r2x_rise', 'r2x_cp' and 'diagnostics' (CPD component collinearity).
   :rtype: dict

.. py:function:: calculate_interaction_tensor(adata: anndata.AnnData, lr_pairs: pd.DataFrame, complex_sep: str | None = "&", communication_score: str = "expression_product", hill_k: float = 0.5)
   :module: cccrise.ccc_rise

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
   :param communication_score: Scoring strategy for communication ('expression_product', 'expression_mean', 'expression_gmean', or 'expression_hill').
   :type communication_score: str, optional
   :param hill_k: Half-maximal constant when using Hill kinetics scoring (default: 0.5).
   :type hill_k: float, optional
   :returns: Tuple containing the interaction tensor with dimensions (n_conditions × rise_rank × rise_rank × n_lr_pairs) and the filtered ligand-receptor pairs DataFrame.
   :rtype: tuple[np.ndarray, pd.DataFrame]


Analysis Functions
~~~~~~~~~~~~~~~~~~

.. py:function:: expression_product_matrix(X1: anndata.AnnData, X2: anndata.AnnData, ligand: str, receptor: str)
   :module: cccrise.utils

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
   :module: cccrise.utils

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

Association Testing
~~~~~~~~~~~~~~~~~~~

One-way ANOVA tests of each CPD component's condition loadings against sample metadata (for example, ALAD status or 6-month outcome).

.. py:function:: run_saved_factors_anova(factors: str | Path | xr.Dataset | anndata.AnnData, adata: anndata.AnnData | None = None, sample_to_group: pd.Series | None = None, group_column: str | None = None, condition_column: str | None = None, control_label: str | None = None, other_label: str | None = None, levels: Sequence[str] | None = None, mapping: Mapping[str, str] | None = None, alpha: float = 0.05, correction: str | None = "fdr_bh", association_name: str | None = None, read_depth_correct: bool = True)
   :module: cccrise.anova

   Run a one-way ANOVA of every CPD component on a saved factors dataset (e.g. ``bal_alad_factors.nc``).

   :param factors: Path to a saved NetCDF factor dataset, an xarray Dataset, or an AnnData holding the factors in ``uns``.
   :param adata: AnnData with the condition and group annotations in ``obs``.
   :param sample_to_group: Pre-extracted condition-to-group mapping, used instead of ``group_column``.
   :param group_column: Phenotype group column in ``adata.obs``.
   :param condition_column: Sample/condition column in ``adata.obs``.
   :param control_label: With ``other_label``, collapses ``group_column`` into two levels.
   :param levels: Subset of group levels to retain.
   :param mapping: Dictionary mapping original group levels to new categories.
   :param alpha: Significance threshold for the ``Significant`` column.
   :param correction: Multiple testing correction across components ('fdr_bh', 'bonferroni', 'holm' or None).
   :param association_name: Name to record in an ``Association`` column.
   :param read_depth_correct: Test the read-depth-corrected condition loadings; this needs the single-cell data in ``adata``. False tests ``uns["A"]`` as is.
   :returns: One row per component with ``F``, ``P``, ``PAdj``, ``EtaSquared``, ``GroupMeanDiff`` and ``Significant``.
   :rtype: pd.DataFrame

.. py:function:: run_multiple_factors_anova(factors: str | Path | xr.Dataset | anndata.AnnData, associations: Sequence[dict[str, Any] | str], adata: anndata.AnnData | None = None, condition_column: str | None = None, alpha: float = 0.05, correction: str | None = "fdr_bh", global_correction: bool = False, read_depth_correct: bool = True)
   :module: cccrise.anova

   Run several ``run_saved_factors_anova`` tests on one factors dataset. Each association is a dictionary (keys such as ``name``, ``group_column``, ``levels``, ``mapping``, ``control_label``, ``other_label``) or the name of an ``adata.obs`` column to test across all its levels. With ``global_correction``, the correction is applied across all associations and components together. Returns one combined DataFrame with an ``Association`` column.

.. py:function:: prepare_anova_inputs(adata: anndata.AnnData, factors: list[np.ndarray], condition_column: str, group_column: str, control_label: str | None = None, other_label: str | None = None, levels: Sequence[str] | None = None, mapping: Mapping[str, str] | None = None)
   :module: cccrise.anova

   Attach one CPD fit's factors [A, B, C, D] to a copy of ``adata`` (with ``uns["A"]`` read-depth corrected) and derive the group label of each condition. Returns the AnnData copy and the group labels aligned to the rows of ``uns["A"]``.

.. py:function:: prepare_group_labels(sample_to_group: pd.Series, levels: Sequence[str] | None = None, mapping: Mapping[str, str] | None = None, control_label: str | None = None, other_label: str | None = None, condition_order: Sequence[str] | None = None)
   :module: cccrise.anova

   Filter, recode, or collapse a condition-to-group mapping: ``levels`` keeps a subset (the other conditions become NaN), ``mapping`` renames levels, and ``control_label`` with ``other_label`` collapses the groups into two. Returns the labels aligned to ``condition_order``.

.. py:function:: anova_component_associations(X: anndata.AnnData, sample_to_group: pd.Series, alpha: float = 0.05, control_label: str | None = None, correction: str | None = None, min_group_size: int = 2)
   :module: cccrise.anova

   One-way ANOVA of each CPD component's condition loading (``X.uns["A"]``) against the group label of each condition. A component's test is NaN when any group has fewer than ``min_group_size`` conditions. Returns one row per component (1-indexed) with ``Component``, ``F``, ``P``, ``PAdj``, ``EtaSquared``, ``GroupMeanDiff``, ``Significant``, ``NGroups`` and ``GroupSizes``.

.. py:function:: adjust_pvalues(p_values: Sequence[float] | np.ndarray, method: str | None = "fdr_bh")
   :module: cccrise.anova

   Adjust p-values for multiple testing across components: Benjamini-Hochberg ('fdr_bh'), Benjamini-Yekutieli ('fdr_by'), Bonferroni ('bonferroni'), Holm ('holm'), or None for the raw values. NaNs for non-finite inputs are preserved.

Pseudobulk Baseline
~~~~~~~~~~~~~~~~~~~

.. py:function:: run_pseudobulk_workflow(adata: anndata.AnnData, cp_rank: int = 8, lr_pairs: pd.DataFrame | None = None, condition_column: str = "sample", groupby: str = "celltype", type: str = "mean", n_iter_max: int = 1000, tol: float = 1e-11, random_state: int | None = None, complex_sep: str | None = "&")
   :module: cccrise.tensor

   Aggregate single cells into pseudobulk matrices per (condition, ``groupby``), compute the four-dimensional communication tensor, fit a CP decomposition, and store the factors, weights and LR pairs in ``adata.uns``.

   :param cp_rank: Rank of the CP decomposition.
   :param lr_pairs: Ligand-receptor pairs; loaded automatically if None.
   :param groupby: Column in ``adata.obs`` defining the cell types.
   :param type: Aggregation method, 'mean' or 'sum'.
   :returns: The updated AnnData object and the R²X.
   :rtype: tuple[anndata.AnnData, float]

Utilities
~~~~~~~~~

.. py:function:: clean_memory()
   :module: cccrise.factors

   Garbage collect and release CuPy device and pinned host memory pools.

``cccrise.ALAD_FACTORS_PATH`` is the path of the canonical ALAD factors file (``cccrise/data/factors/bal_alad_factors.nc``).

Command-Line Runners
--------------------

Each runner is installed as a console script and run with ``uv run <script>``. Runners that take ``--dataset`` use the datasets registered in ``cccrise.runners.datasets.DATASETS`` (``alad`` and ``ibd``) and write their results, with a provenance file, under ``cccrise/data/``.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Script
     - Module
     - Purpose
   * - ``cccrise-run``
     - ``cccrise.runners.run_ccc_rise``
     - The end-to-end cccRISE pipeline; saves the dataset's canonical factors file.
   * - ``cccrise-riserank``
     - ``cccrise.runners.run_rise_rank``
     - RISE (PARAFAC2) rank selection by bi-cross-validation.
   * - ``cccrise-cpdrank``
     - ``cccrise.runners.run_cpd_rank``
     - CPD rank selection on the interaction tensor.
   * - ``cccrise-reduced-rise``
     - ``cccrise.runners.run_reduced_rise``
     - cccRISE refit at a lower RISE rank, for the label-sensitivity panels.
   * - ``cccrise-stability``
     - ``cccrise.runners.run_stability``
     - Per-component stability under patient jackknifing and seed refitting.
   * - ``cccrise-significance``
     - ``cccrise.runners.run_significance``
     - Cell-permutation significance testing of communication.
   * - ``cccrise-nichenet``
     - ``cccrise.runners.run_nichenet``
     - NicheNet-style downstream-target validation of CPD components.
   * - ``cccrise-altcommscore``
     - ``cccrise.runners.run_alt_comm_score``
     - CPD factors for all four communication scores, sharing one RISE fit.
   * - ``cccrise-cohort``
     - ``cccrise.runners.run_cohort``
     - Per-sample and per-cell-type summaries of the cohort a fit was made on.
   * - ``cccrise-alad-programs``, ``cccrise-ibd-programs``
     - ``cccrise.runners.run_alad_programs``, ``cccrise.runners.run_ibd_programs``
     - Tables behind the ALAD and IBD program analyses, from the finalized fit.
   * - ``cccrise-comparators``
     - ``cccrise.runners.run_comparators``
     - Rank selection and fits of the annotation-pooled comparator methods.
   * - ``cccrise-comparator-scores``
     - ``cccrise.runners.run_comparator_scores``
     - How cccRISE's components compare with the comparators'.
   * - ``cccrise-comparator-nichenet``
     - ``cccrise.runners.run_comparator_nichenet``
     - NicheNet tests on cccRISE and every comparator's ligand-receptor columns.
   * - ``cccrise-niches``
     - ``cccrise.runners.run_niches``
     - NICHES on a dataset's production fit, compared with cccRISE's populations.
   * - ``cccrise-scriabin-seeds``, ``cccrise-scriabin-reproduction``
     - ``cccrise.runners.run_scriabin_seeds``, ``cccrise.runners.run_scriabin_reproduction``
     - Scriabin reproducibility analyses.
   * - ``cccrise-fbuild``
     - ``cccrise.figures.common``
     - Build the manuscript figures.
   * - ``cccrise-sourcedata``
     - ``cccrise.figures.source_data``
     - Build the source data files for the figures.
