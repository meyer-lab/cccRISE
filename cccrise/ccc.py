import itertools
from collections import Counter

import numpy as np
from scipy.stats.mstats import gmean
from tensorly.cp_tensor import cp_to_tensor

"""
This file is entirely methods taken from Tensor Cell2Cell
"""


def aggregate_ccc_matrices(ccc_matrices, method="gmean"):
    """Aggregates matrices of communication scores. Each
    matrix has the communication scores across all pairs
    of cell-types/tissues/samples for a different
    pair of interacting proteins.

    Parameters
    ----------
    ccc_matrices : list
        List of matrices of communication scores. Each matrix
        is for an specific pair of interacting proteins.

    method : str, default='gmean'.
        Method to aggregate the matrices element-wise.
        Options are:

        - 'gmean' : Geometric mean in an element-wise way.
        - 'sum' : Sum in an element-wise way.
        - 'mean' : Mean in an element-wise way.

    Returns
    -------
    aggregated_ccc_matrix : numpy.array
        A matrix contiaining aggregated communication scores
        from multiple PPIs. It's shape is of MxM, where M are all
        cell-types/tissues/samples. In directed interactions, the
        vertical axis (axis 0) represents the senders, while the
        horizontal axis (axis 1) represents the receivers.
    """
    if method == "gmean":
        aggregated_ccc_matrix = gmean(ccc_matrices)
    elif method == "sum":
        aggregated_ccc_matrix = np.nansum(ccc_matrices, axis=0)
    elif method == "mean":
        aggregated_ccc_matrix = np.nanmean(ccc_matrices, axis=0)
    else:
        raise ValueError("Not a valid method")

    return aggregated_ccc_matrix


def combine_communication_score(
    raw_product, raw_sum, communication_score="expression_product", hill_k=0.5
):
    """Combine a ligand/receptor pair's raw elementwise product and sum into
    a communication score. Shared by `compute_ccc_matrix` (a single LR pair,
    combined via an outer product over cells/eigen-states) and
    `_fast_interaction_tensor` (many LR pairs combined at once, vectorized).
    `raw_product`/`raw_sum` must already be broadcast to the same target
    shape (e.g. via `np.outer`/`np.add.outer` for a single pair, or the
    matching einsum/broadcast for a batch of pairs).

    Parameters
    ----------
    raw_product : numpy.array
        Elementwise ligand x receptor product, already broadcast to the
        target shape.
    raw_sum : numpy.array
        Elementwise ligand + receptor sum, already broadcast to the same
        shape as `raw_product`.
    communication_score : str, default='expression_product'
        Scoring function for computing the communication score. Options are:

        - 'expression_product' : Multiplication between the expression
            of the interacting proteins.
        - 'expression_mean' : Average between the expression of the
            interacting proteins, as used by CellPhoneDB.
        - 'expression_gmean' : Geometric mean between the expression of the
            interacting proteins.
        - 'expression_hill' : CellChat-style saturating Hill function,
            `product / (hill_k + product)`.

        Every option other than 'expression_product' clips the raw product
        at 0 before use, since they assume non-negative expression (as
        CellChat/CellPhoneDB do) -- inputs derived from signed PARAFAC2/RISE
        eigen-state loadings should be re-centered on real expression first
        (see issue #139) rather than relying on this clip to fix them up.
    hill_k : float, default=0.5
        Half-saturation constant for 'expression_hill'. CellChat's default
        of 0.5 assumes expression normalized to roughly [0, 1]; tune to the
        scale of the input expression values otherwise.

    Returns
    -------
    numpy.array
        Combined communication score, same shape as `raw_product`.
    """
    if communication_score == "expression_product":
        return raw_product
    elif communication_score == "expression_mean":
        return raw_sum / 2.0
    elif communication_score == "expression_gmean":
        return np.sqrt(np.clip(raw_product, 0.0, None))
    elif communication_score == "expression_hill":
        product = np.clip(raw_product, 0.0, None)
        return product / (hill_k + product)
    else:
        raise ValueError(f"Not a valid communication_score: {communication_score!r}")


def compute_ccc_matrix(
    prot_a_exp, prot_b_exp, communication_score="expression_product", hill_k=0.5
):
    """Computes communication scores for an specific
    protein-protein interaction using vectors of gene expression
    levels for a given interacting protein produced by
    different cell-types/tissues/samples.

    Parameters
    ----------
    prot_a_exp : array-like
        Vector with gene expression levels for an interacting protein A
        in a given PPI. Coordinates are different cell-types/tissues/samples.

    prot_b_exp : array-like
        Vector with gene expression levels for an interacting protein B
        in a given PPI. Coordinates are different cell-types/tissues/samples.

    communication_score : str, default='expression_product'
        Scoring function for computing the communication score. See
        `combine_communication_score` for the available options.

    hill_k : float, default=0.5
        Half-saturation constant, used only when
        `communication_score='expression_hill'`. See
        `combine_communication_score`.

    Returns
    -------
    communication_scores : numpy.array
        Matrix MxM, representing the CCC scores of an specific PPI
        across all pairs of cell-types/tissues/samples. M are all
        cell-types/tissues/samples. In directed interactions, the
        vertical axis (axis 0) represents the senders, while the
        horizontal axis (axis 1) represents the receivers.
    """
    raw_product = np.outer(prot_a_exp, prot_b_exp)
    raw_sum = np.add.outer(prot_a_exp, prot_b_exp)
    return combine_communication_score(
        raw_product, raw_sum, communication_score=communication_score, hill_k=hill_k
    )


def get_genes_from_complexes(ppi_data, complex_sep="&", interaction_columns=("A", "B")):
    """
    Gets protein/gene names for individual proteins (subunits when in complex)
    in a list of PPIs. If protein is a complex, for example ProtA&ProtB, it will
    return ProtA and ProtB separately.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    Returns
    -------
    col_a_genes : list
        List of protein/gene names for proteins and subunits in the first column
        of interacting partners.

    complex_a : list
        List of list of subunits of each complex that were present in the first
        column of interacting partners and that were returned as subunits in the
        previous list.

    col_b_genes : list
        List of protein/gene names for proteins and subunits in the second column
        of interacting partners.

    complex_b : list
        List of list of subunits of each complex that were present in the second
        column of interacting partners and that were returned as subunits in the
        previous list.

    complexes : dict
        Dictionary where keys are the complex names in the list of PPIs, while
        values are list of subunits for the respective complex names.
    """
    col_a = interaction_columns[0]
    col_b = interaction_columns[1]

    col_a_genes = set()
    col_b_genes = set()

    complexes = {}
    complex_a = set()
    complex_b = set()
    for _, row in ppi_data.iterrows():
        prot_a = row[col_a]
        prot_b = row[col_b]

        if complex_sep in prot_a:
            comp = {l for l in prot_a.split(complex_sep)}
            complexes[prot_a] = comp
            complex_a = complex_a.union(comp)
        else:
            col_a_genes.add(prot_a)

        if complex_sep in prot_b:
            comp = {r for r in prot_b.split(complex_sep)}
            complexes[prot_b] = comp
            complex_b = complex_b.union(comp)
        else:
            col_b_genes.add(prot_b)

    return col_a_genes, complex_a, col_b_genes, complex_b, complexes


def filter_complex_ppi_by_proteins(
    ppi_data,
    proteins,
    complex_sep="&",
    upper_letter_comparison=True,
    interaction_columns=("A", "B"),
):
    """
    Filters a list of protein-protein interactions that for sure contains
    protein complexes to contain only interacting proteins or subunites
    in a list of specific protein or gene names.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    proteins : list
        A list of protein names to filter PPIs.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    upper_letter_comparison : boolean, default=True
        Whether making uppercase the protein names in the list of proteins and
        the names in the ppi_data to match their names and integrate their
        Useful when there are inconsistencies in the names that comes from a
        expression matrix and from ligand-receptor annotations.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    Returns
    -------
    integrated_ppi : pandas.DataFrame
        A filtered list of PPIs, containing protein complexes in some cases,
        by a given list of proteins or gene names.
    """
    col_a = interaction_columns[0]
    col_b = interaction_columns[1]

    integrated_ppi = ppi_data.copy()

    if upper_letter_comparison:
        integrated_ppi[col_a] = integrated_ppi[col_a].apply(lambda x: str(x).upper())
        integrated_ppi[col_b] = integrated_ppi[col_b].apply(lambda x: str(x).upper())
        prots = {str(p).upper() for p in proteins}
    else:
        prots = set(proteins)

    col_a_genes, complex_a, col_b_genes, complex_b, complexes = (
        get_genes_from_complexes(
            ppi_data=integrated_ppi,
            complex_sep=complex_sep,
            interaction_columns=interaction_columns,
        )
    )

    shared_a_genes = set(col_a_genes & prots)
    shared_b_genes = set(col_b_genes & prots)

    shared_a_complexes = set(complex_a & prots)
    shared_b_complexes = set(complex_b & prots)

    integrated_a_complexes = set()
    integrated_b_complexes = set()
    for k, v in complexes.items():
        if all(p in shared_a_complexes for p in v):
            integrated_a_complexes.add(k)
        elif all(p in shared_b_complexes for p in v):
            integrated_b_complexes.add(k)

    integrated_a = shared_a_genes.union(integrated_a_complexes)
    integrated_b = shared_b_genes.union(integrated_b_complexes)

    filter = (integrated_ppi[col_a].isin(integrated_a)) & (
        integrated_ppi[col_b].isin(integrated_b)
    )
    integrated_ppi = ppi_data.loc[filter].reset_index(drop=True)

    return integrated_ppi


def filter_ppi_by_proteins(
    ppi_data,
    proteins,
    complex_sep=None,
    upper_letter_comparison=True,
    interaction_columns=("A", "B"),
):
    """
    Filters a list of protein-protein interactions to contain
    only interacting proteins in a list of specific protein or gene names.

    Parameters
    ----------
    ppi_data : pandas.DataFrame
        List of protein-protein interactions (or ligand-receptor pairs) used
        for inferring the cell-cell interactions and communication.

    proteins : list
        A list of protein names to filter PPIs.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    upper_letter_comparison : boolean, default=True
        Whether making uppercase the protein names in the list of proteins and
        the names in the ppi_data to match their names and integrate their
        Useful when there are inconsistencies in the names that comes from a
        expression matrix and from ligand-receptor annotations.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a
        dataframe of protein-protein interactions. If the list is for
        ligand-receptor pairs, the first column is for the ligands and the second
        for the receptors.

    Returns
    -------
    integrated_ppi : pandas.DataFrame
        A filtered list of PPIs by a given list of proteins or gene names.
    """

    col_a = interaction_columns[0]
    col_b = interaction_columns[1]

    integrated_ppi = ppi_data.copy()

    if upper_letter_comparison:
        integrated_ppi[col_a] = integrated_ppi[col_a].apply(lambda x: str(x).upper())
        integrated_ppi[col_b] = integrated_ppi[col_b].apply(lambda x: str(x).upper())
        prots = {str(p).upper() for p in proteins}
    else:
        prots = set(proteins)

    if complex_sep is not None:
        integrated_ppi = filter_complex_ppi_by_proteins(
            ppi_data=integrated_ppi,
            proteins=prots,
            complex_sep=complex_sep,
            upper_letter_comparison=False,  # Because it was ran above
            interaction_columns=interaction_columns,
        )
    else:
        integrated_ppi = integrated_ppi[
            (integrated_ppi[col_a].isin(prots)) & (integrated_ppi[col_b].isin(prots))
        ]
    integrated_ppi = integrated_ppi.reset_index(drop=True)
    return integrated_ppi


def get_element_abundances(element_lists):
    """Computes the fraction of occurrence of each element
    in a list of lists.

    Parameters
    ----------
    element_lists : list
        List of lists of elements. Elements will be
        counted only once in each of the lists.

    Returns
    -------
    abundance_dict : dict
        Dictionary containing the number of times that an
        element was present, divided by the total number of
        lists in `element_lists`.
    """
    abundance_dict = Counter(itertools.chain(*map(set, element_lists)))
    total = len(element_lists)
    abundance_dict = {k: v / total for k, v in abundance_dict.items()}
    return abundance_dict


def get_elements_over_fraction(abundance_dict, fraction):
    """Obtains a list of elements with the
    fraction of occurrence at least the threshold.

    Parameters
    ----------
    abundance_dict : dict
        Dictionary containing the number of times that an
        element was present, divided by the total number of
        possible occurrences.

    fraction : float
        Threshold to filter the elements. Elements with at least
        this threshold will be included.

    Returns
    -------
    elements : list
        List of elements that met the fraction criteria.
    """
    elements = [k for k, v in abundance_dict.items() if v >= fraction]
    return elements


def _resolve_tensor_genes_and_cells(how, df_idxs, df_cols, outer_fraction):
    """Determine the gene and cell sets to include in the tensor for a given `how`."""

    def outer_elements(element_lists):
        return set(
            get_elements_over_fraction(
                abundance_dict=get_element_abundances(element_lists=element_lists),
                fraction=outer_fraction,
            )
        )

    if how == "inner":
        return set.intersection(*map(set, df_idxs)), set.intersection(
            *map(set, df_cols)
        )
    if how == "outer":
        return outer_elements(df_idxs), outer_elements(df_cols)
    if how == "outer_genes":
        return outer_elements(df_idxs), set.intersection(*map(set, df_cols))
    if how == "outer_cells":
        return set.intersection(*map(set, df_idxs)), outer_elements(df_cols)
    raise ValueError(
        'Provide a valid way to build the tensor; "how" must be "inner", "outer", "outer_genes" or "outer_cells"'
    )


def _ppi_names(ppi_data, group_ppi_by, interaction_columns):
    """PPI-group names when grouping, else individual "ligand^receptor" names."""
    if group_ppi_by is not None:
        return [group for group, _ in ppi_data.groupby(group_ppi_by)]
    return [
        row[interaction_columns[0]] + "^" + row[interaction_columns[1]]
        for _, row in ppi_data.iterrows()
    ]


def _stack_context_slabs(slabs, n_contexts):
    """Fill a preallocated (context, ...) array one slab at a time."""
    tensors = None
    for i, slab in enumerate(slabs):
        if tensors is None:
            tensors = np.empty((n_contexts, *slab.shape), dtype=slab.dtype)
        tensors[i] = slab
    if tensors is None:
        raise ValueError("No contexts to build a communication tensor from.")
    return tensors


def build_context_ccc_tensor(
    rnaseq_matrices,
    ppi_data,
    how="inner",
    outer_fraction=0.0,
    communication_score="expression_product",
    complex_sep=None,
    upper_letter_comparison=True,
    interaction_columns=("A", "B"),
    group_ppi_by=None,
    group_ppi_method="gmean",
    verbose=True,
    hill_k=0.5,
):
    """Builds a 4D-Communication tensor.
    Takes the gene expression matrices and the list of PPIs to compute
    the communication scores between the interacting cells for each PPI.
    This is done for each context.

    Parameters
    ----------
    rnaseq_matrices : list
        A list with dataframes of gene expression wherein the rows are the genes and
        columns the cell types, tissues or samples.

    ppi_data : pandas.DataFrame
        A dataframe containing protein-protein interactions (rows). It has to
        contain at least two columns, one for the first protein partner in the
        interaction as well as the second protein partner.

    how : str, default='inner'
        Approach to consider cell types and genes present across multiple contexts.

        - 'inner' : Considers only cell types and genes that are present in all
                    contexts (intersection).
        - 'outer' : Considers all cell types and genes that are present
                    across contexts (union).
        - 'outer_genes' : Considers only cell types that are present in all
                          contexts (intersection), while all genes that are
                          present across contexts (union).
        - 'outer_cells' : Considers only genes that are present in all
                          contexts (intersection), while all cell types that are
                          present across contexts (union).

    outer_fraction : float, default=0.0
        Threshold to filter the elements when `how` includes any outer option.
        Elements with a fraction abundance across samples (in `rnaseq_matrices`)
        at least this threshold will be included. When this value is 0, considers
        all elements across the samples. When this value is 1, it acts as using
        `how='inner'`.

    communication_score : str, default='expression_mean'
        Type of communication score to infer the potential use of a given ligand-
        receptor pair by a pair of cells/tissues/samples.
        Available communication_scores are:

        - 'expression_mean' : Computes the average between the expression of a ligand
                              from a sender cell and the expression of a receptor on a
                              receiver cell.
        - 'expression_product' : Computes the product between the expression of a
                                ligand from a sender cell and the expression of a
                                receptor on a receiver cell.
        - 'expression_gmean' : Computes the geometric mean between the expression
                               of a ligand from a sender cell and the
                               expression of a receptor on a receiver cell.
        - 'expression_hill' : Computes a CellChat-style saturating Hill function
                              of the product, `product / (hill_k + product)`.

    hill_k : float, default=0.5
        Half-saturation constant, used only when
        `communication_score='expression_hill'`.

    complex_sep : str, default=None
        Symbol that separates the protein subunits in a multimeric complex.
        For example, '&' is the complex_sep for a list of ligand-receptor pairs
        where a protein partner could be "CD74&CD44".

    upper_letter_comparison : boolean, default=True
        Whether making uppercase the gene names in the expression matrices and the
        protein names in the ppi_data to match their names and integrate their
        respective expression level. Useful when there are inconsistencies in the
        names between the expression matrix and the ligand-receptor annotations.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a dataframe of
        protein-protein interactions. If the list is for ligand-receptor pairs, the
        first column is for the ligands and the second for the receptors.

    group_ppi_by : str, default=None
        Column name in the list of PPIs used for grouping individual PPIs into major
        groups such as signaling pathways.

    group_ppi_method : str, default='gmean'
        Method for aggregating multiple PPIs into major groups.

        - 'mean' : Computes the average communication score among all PPIs of the
                   group for a given pair of cells/tissues/samples
        - 'gmean' : Computes the geometric mean of the communication scores among all
                    PPIs of the group for a given pair of cells/tissues/samples
        - 'sum' : Computes the sum of the communication scores among all PPIs of the
                  group for a given pair of cells/tissues/samples

    verbose : boolean, default=False
            Whether printing or not steps of the analysis.

    Returns
    -------
    tensors : list
        List of 3D-Communication tensors for each context. This list corresponds to
        the 4D-Communication tensor.

    genes : list
        List of genes included in the tensor.

    cells : list
        List of cells included in the tensor.

    ppi_names: list
        List of names for each of the PPIs included in the tensor. Used as labels for the
        elements in the cognate tensor dimension (in the attribute order_names of the
        InteractionTensor)

    mask_tensor: numpy.array
        Mask used to exclude values in the tensor. When using how='outer' it masks
        missing values (e.g., cell types that are not present in a given context),
        while using how='inner' makes the mask_tensor to be None.
    """
    df_idxs = [list(rnaseq.index) for rnaseq in rnaseq_matrices]
    df_cols = [list(rnaseq.columns) for rnaseq in rnaseq_matrices]

    genes, cells = _resolve_tensor_genes_and_cells(
        how, df_idxs, df_cols, outer_fraction
    )

    # Preserve order or sort new set (either inner or outer)
    genes = df_idxs[0] if set(df_idxs[0]) == genes else sorted(genes)

    cells = df_cols[0] if set(df_cols[0]) == cells else sorted(cells)

    # Filter PPI data for
    ppi_data_ = filter_ppi_by_proteins(
        ppi_data=ppi_data,
        proteins=genes,
        complex_sep=complex_sep,
        upper_letter_comparison=upper_letter_comparison,
        interaction_columns=interaction_columns,
    )

    if verbose:
        print("Building tensor for the provided context")

    def context_tensor(rnaseq):
        tensor = generate_ccc_tensor(
            rnaseq_data=rnaseq.reindex(genes).reindex(cells, axis="columns"),
            ppi_data=ppi_data_,
            communication_score=communication_score,
            interaction_columns=interaction_columns,
            hill_k=hill_k,
        )
        if group_ppi_by is not None:
            tensor = aggregate_ccc_tensor(
                ccc_tensor=tensor,
                ppi_data=ppi_data_,
                group_ppi_by=group_ppi_by,
                group_ppi_method=group_ppi_method,
            )
        return np.asarray(tensor, dtype=float)

    ppi_names = _ppi_names(ppi_data_, group_ppi_by, interaction_columns)
    tensors = _stack_context_slabs(
        (context_tensor(rnaseq) for rnaseq in rnaseq_matrices), len(rnaseq_matrices)
    )

    # Generate mask:
    if how != "inner":
        mask_tensor = (~np.isnan(tensors)).astype(int)
    else:
        mask_tensor = None
    # In place, `tensors` is this function's own array, so the default copy
    # would duplicate the whole thing.
    tensors = np.nan_to_num(tensors, copy=False)
    return tensors, genes, cells, ppi_names, mask_tensor


def generate_ccc_tensor(
    rnaseq_data,
    ppi_data,
    communication_score="expression_product",
    interaction_columns=("A", "B"),
    hill_k=0.5,
):
    """Computes a 3D-Communication tensor for a given context based on the gene
    expression matrix and the list of PPIS

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression matrix for a given context, sample or condition. Rows are
        genes and columns are cell types/tissues/samples.

    ppi_data : pandas.DataFrame
        A dataframe containing protein-protein interactions (rows). It has to
        contain at least two columns, one for the first protein partner in the
        interaction as well as the second protein partner.

    communication_score : str, default='expression_mean'
        Type of communication score to infer the potential use of a given ligand-
        receptor pair by a pair of cells/tissues/samples.
        Available communication_scores are:

        - 'expression_mean' : Computes the average between the expression of a ligand
                              from a sender cell and the expression of a receptor on a
                              receiver cell.
        - 'expression_product' : Computes the product between the expression of a
                                ligand from a sender cell and the expression of a
                                receptor on a receiver cell.
        - 'expression_gmean' : Computes the geometric mean between the expression
                               of a ligand from a sender cell and the
                               expression of a receptor on a receiver cell.
        - 'expression_hill' : Computes a CellChat-style saturating Hill function
                              of the product, `product / (hill_k + product)`.

    hill_k : float, default=0.5
        Half-saturation constant, used only when
        `communication_score='expression_hill'`.

    interaction_columns : tuple, default=('A', 'B')
        Contains the names of the columns where to find the partners in a dataframe of
        protein-protein interactions. If the list is for ligand-receptor pairs, the
        first column is for the ligands and the second for the receptors.

    Returns
    -------
    ccc_tensor : ndarray list
        List of directed cell-cell communication matrices, one for each ligand-
        receptor pair in ppi_data. These matrices contain the communication score for
        pairs of cells for the corresponding PPI. This tensor represent a
        3D-communication tensor for the context.
    """
    ppi_a = interaction_columns[0]
    ppi_b = interaction_columns[1]

    # Convert DataFrame to NumPy arrays for faster indexing
    gene_index = {gene: i for i, gene in enumerate(rnaseq_data.index)}
    data_array = rnaseq_data.values  # Convert to NumPy once

    ccc_tensor = []
    for _, ppi in ppi_data.iterrows():
        # Use NumPy indexing instead of pandas .loc
        gene_a_idx = gene_index[ppi[ppi_a]]
        gene_b_idx = gene_index[ppi[ppi_b]]

        v = data_array[gene_a_idx, :]
        w = data_array[gene_b_idx, :]

        ccc_tensor.append(
            compute_ccc_matrix(
                prot_a_exp=v,
                prot_b_exp=w,
                communication_score=communication_score,
                hill_k=hill_k,
            ).tolist()
        )
    return ccc_tensor


def aggregate_ccc_tensor(
    ccc_tensor, ppi_data, group_ppi_by=None, group_ppi_method="gmean"
):
    """Aggregates communication scores of multiple PPIs into major groups
    (e.g., pathways) in a communication tensor

    Parameters
    ----------
    ccc_tensor : ndarray list
        List of directed cell-cell communication matrices, one for each ligand-
        receptor pair in ppi_data. These matrices contain the communication score for
        pairs of cells for the corresponding PPI. This tensor represent a
        3D-communication tensor for the context.

    ppi_data : pandas.DataFrame
        A dataframe containing protein-protein interactions (rows). It has to
        contain at least two columns, one for the first protein partner in the
        interaction as well as the second protein partner.

    group_ppi_by : str, default=None
        Column name in the list of PPIs used for grouping individual PPIs into major
        groups such as signaling pathways.

    group_ppi_method : str, default='gmean'
        Method for aggregating multiple PPIs into major groups.

        - 'mean' : Computes the average communication score among all PPIs of the
                   group for a given pair of cells/tissues/samples
        - 'gmean' : Computes the geometric mean of the communication scores among all
                    PPIs of the group for a given pair of cells/tissues/samples
        - 'sum' : Computes the sum of the communication scores among all PPIs of the
                  group for a given pair of cells/tissues/samples

    Returns
    -------
    aggregated_tensor : ndarray list
        List of directed cell-cell communication matrices, one for each major group of
        ligand-receptor pair in ppi_data. These matrices contain the communication
        score for pairs of cells for the corresponding PPI group. This tensor
        represent a 3D-communication tensor for the context, but for major groups
        instead of individual PPIs.
    """
    tensor_ = np.array(ccc_tensor)
    aggregated_tensor = []
    for _, df in ppi_data.groupby(group_ppi_by):
        lr_idx = list(df.index)
        ccc_matrices = tensor_[lr_idx]
        aggregated_tensor.append(
            aggregate_ccc_matrices(
                ccc_matrices=ccc_matrices, method=group_ppi_method
            ).tolist()
        )
    return aggregated_tensor


def calculate_r2x(cp_weights, cp_factors, interaction_tensor):
    """Calculate R2X for the CP decomposition of the interaction tensor"""
    reconstructed = cp_to_tensor((cp_weights, cp_factors))
    total_variance = np.sum(interaction_tensor**2)
    error = np.sum((interaction_tensor - reconstructed) ** 2)
    final_R2X = 1 - (error / total_variance) if total_variance > 0 else 0.0

    return final_R2X
