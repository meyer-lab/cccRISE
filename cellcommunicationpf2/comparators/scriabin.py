"""Python re-implementation of Scriabin (BlishLab/scriabin, R).

See :func:`scriabin`'s docstring for what algorithm this reproduces, how it
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


def scriabin(
    adata: anndata.AnnData,
    send_cells: Sequence[str],
    rec_cells: Sequence[str],
    lr_pairs: pd.DataFrame | None = None,
    complex_sep: str = "&",
    complex_agg_method: str = "min",
) -> pd.DataFrame:
    """Re-implementation of Scriabin's single-cell-resolution edge score.

    This is a from-scratch Python re-implementation of the core scoring loop
    in ``scriabin::GenerateInteractome()`` (BlishLab/scriabin, R),
    reconstructed by reading that function's decompiled body -- it is not a
    wrapper around the R package, and scriabin is not a dependency of this
    project. It is meant to let CCC-RISE be benchmarked against Scriabin's
    single-cell-resolution approach without an R runtime (this project
    previously ran the actual R package via an `R/` comparator harness,
    since removed in favor of this module).

    Upstream computes, for every (sender cell, receiver cell, LR-pair)
    triple, ``edgeweight = log1p(ligand_expr) * log1p(receptor_expr)``,
    keeping only positive edgeweights, via a nested loop over every sender
    and receiver cell. That loop is what makes it O(n_send * n_rec) in
    practice, but the computation itself is a plain elementwise product, so
    it's reproduced here as one vectorized outer product per LR pair rather
    than a loop. Multi-subunit complexes (ligand or receptor made of several
    gene products) are aggregated the same way CCC-RISE aggregates them --
    via :func:`cellcommunicationpf2.ccc_rise.add_complexes_to_expression`
    (default ``min`` across subunits) -- rather than via scriabin's own
    complex handling, so the two methods differ only in how they turn
    per-cell expression into a communication score, not in what counts as a
    ligand/receptor.

    Known differences from upstream:

    - **LR database**: this uses this project's shared, complex-aware LR
      table (:func:`cellcommunicationpf2.import_data.import_ligand_receptor_pairs`,
      the same one CCC-RISE uses) instead of scriabin's own bundled
      database (OmniPath/CellChatDB/etc.), so the comparison isolates the
      algorithmic approach rather than database choice.
    - **Per-cell gene-set flags dropped**: upstream also flags whether each
      ligand/receptor was among a cell's top variable genes
      (``lig_geneset``/``rec_geneset``, driven by an externally-supplied
      ``ranked_genes`` argument used by downstream prioritization steps).
      This project's now-removed R harness (``run_scriabin.R``) already
      supplied an inert placeholder for that argument since it wasn't used
      downstream there either, so it's omitted here too.
    - Downstream steps upstream also ships (binning, ``PrioritizeInteractome``,
      alluvial plots, etc.) aren't reproduced -- only the raw per-cell-pair
      edge score, which is all this project's R harness ever consumed.

    Cost is still O(n_send * n_rec * n_lr_pairs) in memory (one dense
    3D array before filtering to positive edges), so keep ``send_cells``/
    ``rec_cells`` to tens of cells, not thousands, exactly as documented for
    the R version.

    Comparing against CCC-RISE
    ---------------------------
    Scriabin has no fitted model to compare component-for-component -- it's
    a per-cell-pair score, so the reviewer-requested comparison (R1#1, R1#2)
    is a **concordance analysis**, not a factor congruence:

    1. Reconstruct a matching per-cell-pair score from CCC-RISE: combine its
       per-cell RISE projections (``adata.obsm["sc_B"]``/``adata.obsm["rc_C"]``)
       with the CP factors (``B``/``C``/``D``) the same way the full
       communication tensor is reconstructed from CP factors, but evaluated
       at individual sender/receiver cells rather than celltype/condition
       aggregates.
    2. Run both methods on the *same* sampled sender/receiver cells (same
       seed) and the same LR pairs, then compute a Spearman or Pearson
       correlation between this function's ``edgeweight`` column and
       CCC-RISE's reconstructed per-cell-pair score.
    3. Repeat on a single-condition subset (multi-condition modeling turned
       off) to test whether CCC-RISE's concordance with Scriabin degrades --
       the "single-condition degradation test" Reviewer #1 asked for.

    Parameters
    ----------
    adata : anndata.AnnData
        Already-normalized (log-transformed) expression data, matching the
        normalization convention CCC-RISE itself assumes -- this function
        does no normalization of its own.
    send_cells, rec_cells : Sequence[str]
        Sender and receiver cell barcodes (``adata.obs_names`` values) to
        score all pairs of. Keep these small (tens of cells) -- see cost
        note above.
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

    Returns
    -------
    pd.DataFrame
        One row per (sender cell, receiver cell, LR pair) with a positive
        edgeweight, columns ``sender``, ``receiver``, ``lr_pair``,
        ``ligand``, ``receptor``, ``ligand_expression``,
        ``receptor_expression``, ``edgeweight``.
    """
    if lr_pairs is None:
        lr_pairs = import_ligand_receptor_pairs()

    _, _, _, _, complexes = get_genes_from_complexes(
        lr_pairs, complex_sep=complex_sep, interaction_columns=("ligand", "receptor")
    )

    send_expr = add_complexes_to_expression(
        _dense_gene_by_cell(adata, send_cells), complexes, agg_method=complex_agg_method
    )
    rec_expr = add_complexes_to_expression(
        _dense_gene_by_cell(adata, rec_cells), complexes, agg_method=complex_agg_method
    )

    ligand_names = lr_pairs["ligand"].to_numpy()
    receptor_names = lr_pairs["receptor"].to_numpy()
    pair_labels = (
        lr_pairs["interaction_symbol"].to_numpy()
        if "interaction_symbol" in lr_pairs.columns
        else np.array([f"{lg}^{rc}" for lg, rc in zip(ligand_names, receptor_names)])
    )

    lig_mat = send_expr.reindex(ligand_names).fillna(0.0).to_numpy()  # (n_pairs, n_send)
    rec_mat = rec_expr.reindex(receptor_names).fillna(0.0).to_numpy()  # (n_pairs, n_rec)

    lig_log = np.log1p(np.clip(lig_mat, 0, None))
    rec_log = np.log1p(np.clip(rec_mat, 0, None))

    edge = lig_log[:, :, None] * rec_log[:, None, :]  # (n_pairs, n_send, n_rec)
    pair_idx, send_idx, rec_idx = np.nonzero(edge > 0)

    sender_ids = np.asarray(send_expr.columns)
    receiver_ids = np.asarray(rec_expr.columns)

    return pd.DataFrame(
        {
            "sender": sender_ids[send_idx],
            "receiver": receiver_ids[rec_idx],
            "lr_pair": pair_labels[pair_idx],
            "ligand": ligand_names[pair_idx],
            "receptor": receptor_names[pair_idx],
            "ligand_expression": lig_log[pair_idx, send_idx],
            "receptor_expression": rec_log[pair_idx, rec_idx],
            "edgeweight": edge[pair_idx, send_idx, rec_idx],
        }
    )
