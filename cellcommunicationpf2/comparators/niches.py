"""Python re-implementation of NICHES (msraredon/NICHES, R).

See :func:`niches`'s docstring for what algorithm this reproduces, how it
differs from the original R package, and how to compare its output against
CCC-RISE.
"""

from collections.abc import Sequence

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from ..ccc import get_genes_from_complexes
from ..ccc_rise import add_complexes_to_expression
from ..import_data import import_ligand_receptor_pairs


def _dense_gene_by_cell(adata: anndata.AnnData, cell_ids: Sequence[str]) -> pd.DataFrame:
    """Densify ``adata[cell_ids]`` into a ``(genes x cells)`` DataFrame."""
    sub = adata[list(cell_ids), :]
    X = sub.X
    X = X.toarray() if issparse(X) else np.asarray(X)
    return pd.DataFrame(np.asarray(X).T, index=sub.var_names, columns=sub.obs_names)


def niches(
    adata: anndata.AnnData,
    groupby: str,
    lr_pairs: pd.DataFrame | None = None,
    complex_sep: str = "&",
    complex_agg_method: str = "min",
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-implementation of NICHES' cell-to-cell signal matrix.

    This is a from-scratch Python re-implementation of ``NICHES::RunCellToCell()``
    (msraredon/NICHES, R) as called by ``RunNICHES(..., CellToCell = TRUE)`` --
    reconstructed by reading that function's decompiled body -- it is not a
    wrapper around the R package, and NICHES is not a dependency of this
    project. It is meant to let CCC-RISE be benchmarked against NICHES'
    single-cell-resolution approach without an R runtime (see
    ``R/README.md`` for the R comparator harness this replaces). Only the
    ``CellToCell`` mode is reproduced -- ``CellToSystem``/``SystemToCell``/
    spatial-neighborhood modes upstream also offers aren't used by this
    project's R harness either.

    Upstream algorithm: for every pair of cell groups (sender type, receiver
    type -- including a type paired with itself), it builds a per-cell
    ligand-complex expression vector and a per-cell receptor-complex
    expression vector (multi-subunit complexes combined via an elementwise
    product across subunits), then -- rather than exhaustively enumerating
    every (sender cell, receiver cell) combination -- draws a random
    same-size subsample from each type's cell pool (size
    ``min(n_cells_sender_type, n_cells_receiver_type)``) and takes the
    elementwise product of the two subsamples, positionally paired. This
    scales to large datasets by sampling representative pairs per
    (sender-type, receiver-type) combination instead of computing the full
    cross product. This function reproduces that same sampling scheme, with
    ``add_complexes_to_expression`` used for the complex step (matching
    CCC-RISE's own complex handling) instead of NICHES' internal
    subunit-product code.

    Known differences from upstream:

    - **LR database**: this uses this project's shared, complex-aware LR
      table (:func:`cellcommunicationpf2.import_data.import_ligand_receptor_pairs`,
      the same one CCC-RISE uses) instead of NICHES' own bundled "fantom5"
      database, so the comparison isolates the algorithmic approach rather
      than database choice (this project's R harness used NICHES' own
      database instead, since NICHES doesn't directly consume an external
      LR table the way the other R comparators do).
    - **No post-hoc re-normalization**: upstream log-normalizes the
      resulting signal matrix itself (treating it like a second expression
      matrix, via ``Seurat::NormalizeData``) before returning it. This
      returns the raw elementwise-product signal; apply your own
      normalization if you want to match that step.
    - **No ``min.cells.per.ident``/``min.cells.per.gene`` filtering**:
      upstream drops sparsely-represented groups/genes by default; this
      function scores every group and every LR pair in ``lr_pairs`` --
      filter ``adata``/``lr_pairs`` yourself beforehand if you want that
      behavior.

    Comparing against CCC-RISE
    ---------------------------
    Like Scriabin, NICHES has no fitted model to compare component-for-
    component -- it's a per-cell-pair score, so (per Reviewer #1's ask for
    "additional single-cell-resolution concordance") the comparison is the
    same concordance analysis proposed for Scriabin
    (see :mod:`cellcommunicationpf2.comparators.scriabin`):

    1. Reconstruct a matching per-cell-pair score from CCC-RISE by combining
       its per-cell RISE projections (``adata.obsm["sc_B"]``/
       ``adata.obsm["rc_C"]``) with the CP factors (``B``/``C``/``D``),
       evaluated at individual cells rather than celltype/condition
       aggregates.
    2. For the same (sender-type, receiver-type) sampled cell pairs this
       function draws (fix ``random_state`` for reproducibility), compute a
       Spearman/Pearson correlation between this function's signal values
       and CCC-RISE's reconstructed per-cell-pair score, per LR pair or
       pooled across pairs.
    3. Use NICHES' broader per-celltype-pair coverage (every group x group
       combination, not just one sampled sender/receiver group like
       Scriabin) to check whether concordance with CCC-RISE holds uniformly
       across celltype pairs or is concentrated in a few.

    Parameters
    ----------
    adata : anndata.AnnData
        Already-normalized (log-transformed) expression data, matching the
        normalization convention CCC-RISE itself assumes -- this function
        does no normalization of its own.
    groupby : str
        ``adata.obs`` column naming each cell's group (e.g. celltype).
    lr_pairs : pd.DataFrame, optional
        LR pair table with ``ligand``/``receptor`` columns (and, if present,
        ``interaction_symbol`` for pair labels). Defaults to
        ``import_ligand_receptor_pairs()``, the same table CCC-RISE uses.
    complex_sep : str, default "&"
        Subunit separator for multi-gene complexes, matching CCC-RISE's own
        default.
    complex_agg_method : str, default "min"
        How to aggregate subunit expression into a complex's expression --
        passed through to ``add_complexes_to_expression`` (``"min"``,
        ``"mean"``, or ``"gmean"``).
    random_state : int, optional
        Seed for the per-(sender-type, receiver-type) cell subsampling.

    Returns
    -------
    signal_matrix : pd.DataFrame
        LR pair x edge matrix (edge = one sampled sender-receiver cell
        pair), values are the elementwise ligand x receptor product.
    edge_metadata : pd.DataFrame
        One row per edge (matching ``signal_matrix`` columns), with
        ``sending_type``, ``receiving_type``, ``sender_cell``,
        ``receiver_cell``.
    """
    if lr_pairs is None:
        lr_pairs = import_ligand_receptor_pairs()

    _, _, _, _, complexes = get_genes_from_complexes(
        lr_pairs, complex_sep=complex_sep, interaction_columns=("ligand", "receptor")
    )

    ligand_names = lr_pairs["ligand"].to_numpy()
    receptor_names = lr_pairs["receptor"].to_numpy()
    pair_labels = (
        lr_pairs["interaction_symbol"].to_numpy()
        if "interaction_symbol" in lr_pairs.columns
        else np.array([f"{lg}^{rc}" for lg, rc in zip(ligand_names, receptor_names)])
    )

    groups = sorted(adata.obs[groupby].unique())
    cells_by_group = {
        g: adata.obs_names[adata.obs[groupby] == g].to_numpy() for g in groups
    }
    expr_by_group = {
        g: add_complexes_to_expression(
            _dense_gene_by_cell(adata, cells_by_group[g]),
            complexes,
            agg_method=complex_agg_method,
        )
        for g in groups
    }
    ligand_mat_by_group = {
        g: expr_by_group[g].reindex(ligand_names).fillna(0.0).to_numpy()
        for g in groups
    }
    receptor_mat_by_group = {
        g: expr_by_group[g].reindex(receptor_names).fillna(0.0).to_numpy()
        for g in groups
    }

    rng = np.random.default_rng(random_state)

    signal_blocks: list[pd.DataFrame] = []
    meta_blocks: list[pd.DataFrame] = []
    for sender_type in groups:
        sender_pool = cells_by_group[sender_type]
        lig_mat_full = ligand_mat_by_group[sender_type]
        for receiver_type in groups:
            receiver_pool = cells_by_group[receiver_type]
            rec_mat_full = receptor_mat_by_group[receiver_type]

            n_pair_cells = min(len(sender_pool), len(receiver_pool))
            if n_pair_cells == 0:
                continue
            sender_idx = rng.choice(len(sender_pool), size=n_pair_cells, replace=False)
            receiver_idx = rng.choice(
                len(receiver_pool), size=n_pair_cells, replace=False
            )

            signal = lig_mat_full[:, sender_idx] * rec_mat_full[:, receiver_idx]
            sender_cells = sender_pool[sender_idx]
            receiver_cells = receiver_pool[receiver_idx]
            edge_ids = [f"{s}—{r}" for s, r in zip(sender_cells, receiver_cells)]

            signal_blocks.append(pd.DataFrame(signal, index=pair_labels, columns=edge_ids))
            meta_blocks.append(
                pd.DataFrame(
                    {
                        "sending_type": sender_type,
                        "receiving_type": receiver_type,
                        "sender_cell": sender_cells,
                        "receiver_cell": receiver_cells,
                    },
                    index=edge_ids,
                )
            )

    signal_matrix = pd.concat(signal_blocks, axis=1)
    edge_metadata = pd.concat(meta_blocks, axis=0)
    return signal_matrix, edge_metadata
