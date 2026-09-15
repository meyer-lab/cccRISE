"""`cccrise.loading.load_normalized` -- the filter-then-normalize loader.

The load path it replaces (`vsparse.load_and_normalize`) fused filtering and
normalization, so these tests pin two things: that nothing moved numerically
when no gene is excluded, and that excluding a gene really does change the
depth divisor rather than just dropping a column.
"""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import vsparse

from cccrise.gene_filters import describe_mask, mito_ribo_mask
from cccrise.loading import load_normalized

N_CELLS, N_GENES = 300, 80


def _counts() -> np.ndarray:
    """Integer counts with QC-failing cells and two all-zero genes."""
    rng = np.random.default_rng(0)
    x = np.ceil(
        sp.random_array((N_CELLS, N_GENES), density=0.2, rng=rng).toarray() * 40
    )
    x[:5, :] = 0.0  # empty cells
    x[5, :] = 0.0
    x[5, 0] = 3.0  # below min_cell_counts
    x[:, 7] = 0.0  # never-detected genes
    x[:, 8] = 0.0
    # One gene carrying a large share of every cell's depth, so removing it
    # visibly moves the divisor.
    x[:, 1] = 500.0
    return x


def _var() -> pd.DataFrame:
    symbols = [f"G{i}" for i in range(N_GENES)]
    symbols[1] = "MT-CO1"
    symbols[2] = "RPL13"
    symbols[3] = "MRPS12"
    # Real symbols that a naive ^RP[LS]/^MRP[SL] regex would catch but which are
    # not ribosomal at all -- see test_mito_ribo_mask_keeps_the_s6_kinases.
    symbols[4] = "RPS6KA1"
    symbols[5] = "RPS6KB2"
    symbols[6] = "MRPS30-DT"
    # Flags as the Sanger pipeline actually sets them: on the ribosomal proteins
    # themselves, not on everything whose symbol starts with RP. Assigning them
    # by prefix here would make `test_mito_ribo_mask_keeps_the_s6_kinases`
    # vacuous -- the fixture would contain the very bug it guards against.
    ribosomal = {"RPL13", "MRPS12"}
    return pd.DataFrame(
        {
            "gene_symbols": symbols,
            "gene_group__mito_transcript": [s.startswith("MT-") for s in symbols],
            "gene_group__ribo_protein": [s in ribosomal for s in symbols],
        },
        index=[f"ENSG{i:05d}" for i in range(N_GENES)],
    )


@pytest.fixture(scope="module")
def packed_path(tmp_path_factory) -> str:
    """The counts written in the packed on-disk format the loaders require."""
    adata = anndata.AnnData(
        X=sp.csr_array(_counts()),
        obs=pd.DataFrame(
            {"sample": np.resize(["a", "b", "c"], N_CELLS)},
            index=[f"c{i}" for i in range(N_CELLS)],
        ),
        var=_var(),
    )
    path = tmp_path_factory.mktemp("vcs") / "counts.ivcsr.h5ad"
    # VCSR (row-major) in, so the normalized view materializes as CSR out.
    vsparse.VCSCAnnData.from_anndata(adata, format="csr").write_h5ad(
        path, format="ivcsc"
    )
    return str(path)


# -- parity with the load path this replaces ---------------------------------


@pytest.mark.parametrize("gene_threshold", [0.0, 0.001, 0.05])
def test_matches_load_and_normalize_bitwise(packed_path, gene_threshold):
    """With no exclusion this must be the old call, to the bit.

    The IBD rank-selection numbers already on record were produced by
    `load_and_normalize`; if this drifts, they stop being comparable.
    """
    ref = vsparse.load_and_normalize(
        packed_path, min_cell_counts=10.0, gene_threshold=gene_threshold
    )
    new = load_normalized(
        packed_path, min_cell_counts=10.0, gene_threshold=gene_threshold
    )

    assert new.shape == ref.shape
    assert list(new.obs_names) == list(ref.obs_names)
    assert list(new.var_names) == list(ref.var_names)
    assert new.X.dtype == ref.X.dtype
    assert np.array_equal(new.X.toarray(), ref.X.toarray())


def test_matches_load_and_normalize_under_min_cells_and_obs_filter(packed_path):
    """The two filters that change *which* rows the statistics see."""
    kwargs = {"min_cell_counts": 10.0, "gene_threshold": 0.001}
    obs_filter = lambda obs: (obs["sample"] == "a").to_numpy()

    for extra in ({"min_cells": 30}, {"obs_filter": obs_filter}):
        ref = vsparse.load_and_normalize(packed_path, **kwargs, **extra)
        new = load_normalized(packed_path, **kwargs, **extra)
        assert new.shape == ref.shape, extra
        assert np.array_equal(new.X.toarray(), ref.X.toarray()), extra


# -- the reason this loader exists -------------------------------------------


def test_exclusion_changes_the_depth_divisor(packed_path):
    """Excluding before normalizing is not the same as dropping a column after.

    This is the whole point of the migration. `mito_ribo_mask` removes genes
    that carry a large share of each cell's counts, so they must be gone
    *before* the per-cell depth is measured. If these two ever agree, the
    exclusion has silently become post-hoc.
    """
    kwargs = {"min_cell_counts": 10.0, "gene_threshold": 0.001}
    excluded = load_normalized(packed_path, **kwargs, exclude_genes=mito_ribo_mask)
    unfiltered = load_normalized(packed_path, **kwargs)
    post_hoc = unfiltered[:, ~mito_ribo_mask(unfiltered.var)].copy()

    assert excluded.shape == post_hoc.shape
    assert list(excluded.var_names) == list(post_hoc.var_names)
    assert not np.allclose(excluded.X.toarray(), post_hoc.X.toarray())


def test_exclusion_drops_exactly_the_masked_genes(packed_path):
    kwargs = {"min_cell_counts": 10.0, "gene_threshold": 0.001}
    unfiltered = load_normalized(packed_path, **kwargs)
    excluded = load_normalized(packed_path, **kwargs, exclude_genes=mito_ribo_mask)

    removed = set(unfiltered.var_names) - set(excluded.var_names)
    assert removed == set(unfiltered.var_names[mito_ribo_mask(unfiltered.var)])
    assert "MT-CO1" in set(
        unfiltered.var["gene_symbols"][mito_ribo_mask(unfiltered.var)]
    )


def test_qc_thresholds_are_measured_before_exclusion(packed_path):
    """A gene must not survive `gene_threshold` only because a mito gene left.

    Thresholds read raw counts from the unfiltered matrix, so the surviving
    gene set is the unfiltered one minus the mask -- never more.
    """
    kwargs = {"min_cell_counts": 10.0, "gene_threshold": 0.05}
    unfiltered = load_normalized(packed_path, **kwargs)
    excluded = load_normalized(packed_path, **kwargs, exclude_genes=mito_ribo_mask)

    assert set(excluded.var_names) <= set(unfiltered.var_names)
    # Cells are unaffected by a gene mask: the cell filter reads raw totals too.
    assert list(excluded.obs_names) == list(unfiltered.obs_names)


# -- the sparse-plus-means convention ----------------------------------------


def test_means_is_the_column_mean_of_the_returned_matrix(packed_path):
    """`condition_matrices` subtracts `var["means"]`; it must match `X`.

    The recipe computes the centering term but is asked to leave it out of
    `X`, so that the matrix stays sparse and centering happens once a
    condition block is densified.
    """
    adata = load_normalized(packed_path, min_cell_counts=10.0, gene_threshold=0.001)
    expected = np.asarray(adata.X.mean(axis=0)).reshape(-1)
    assert np.allclose(np.asarray(adata.var["means"]), expected)


def test_statistics_are_recomputed_for_the_filtered_subset(packed_path):
    """Indexing marks inherited statistics stale; the loader must not ship those."""
    adata = load_normalized(packed_path, min_cell_counts=10.0, gene_threshold=0.001)
    assert adata.uns["vsparse"]["recipe"] == "parafac2"
    assert adata.uns["vsparse"]["stale"] is False


def test_row_major_input_materializes_as_csr(packed_path):
    """Downstream slices by cell, so the cheap axis has to be rows."""
    adata = load_normalized(packed_path, min_cell_counts=10.0, gene_threshold=0.001)
    assert isinstance(adata.X, sp.csr_array)


def test_dtype_is_selectable_without_changing_values(packed_path):
    kwargs = {"min_cell_counts": 10.0, "gene_threshold": 0.001}
    wide = load_normalized(packed_path, **kwargs, dtype=np.float64)
    narrow = load_normalized(packed_path, **kwargs)
    assert wide.X.dtype == np.float64
    assert narrow.X.dtype == np.float32
    assert np.allclose(wide.X.toarray(), narrow.X.toarray(), atol=1e-6)


# -- input validation --------------------------------------------------------


def test_a_bad_mask_is_rejected_before_the_expensive_work(packed_path):
    kwargs = {"min_cell_counts": 10.0, "gene_threshold": 0.001}
    with pytest.raises(ValueError, match="one-dimensional mask"):
        load_normalized(
            packed_path, **kwargs, exclude_genes=lambda var: np.ones(3, bool)
        )
    with pytest.raises(ValueError, match="boolean mask"):
        load_normalized(
            packed_path, **kwargs, exclude_genes=lambda var: np.ones(len(var), int)
        )


def test_filtering_everything_out_is_an_error_not_an_empty_matrix(packed_path):
    kwargs = {"min_cell_counts": 10.0, "gene_threshold": 0.001}
    with pytest.raises(ValueError, match="every gene"):
        load_normalized(
            packed_path, **kwargs, exclude_genes=lambda var: np.ones(len(var), bool)
        )
    with pytest.raises(ValueError, match="every cell"):
        load_normalized(packed_path, min_cell_counts=1e12, gene_threshold=0.0)
    with pytest.raises(ValueError, match="selected no cells"):
        load_normalized(
            packed_path, **kwargs, obs_filter=lambda obs: np.zeros(len(obs), bool)
        )


# -- the mask itself ---------------------------------------------------------


def test_mito_ribo_mask_separates_the_two_ribosomes():
    """`gene_group__ribo_protein` lumps both ribosomes together; we split them."""
    var = _var()
    both = mito_ribo_mask(var)
    cytosolic_only = mito_ribo_mask(var, exclude_mito_ribosomal=False)
    symbols = var["gene_symbols"]

    assert both[symbols == "MT-CO1"].all()  # mitochondrial transcript
    assert both[symbols == "RPL13"].all()  # cytosolic ribosomal protein
    assert both[symbols == "MRPS12"].all()  # mitochondrial ribosomal protein
    # The mitochondrial ribosome is the separable half.
    assert not cytosolic_only[symbols == "MRPS12"].any()
    assert cytosolic_only[symbols == "RPL13"].all()


def test_describe_mask_names_what_it_removed():
    var = _var()
    assert "nothing excluded" in describe_mask(var, np.zeros(len(var), bool))
    assert "MT-CO1" in describe_mask(var, mito_ribo_mask(var))


def test_mito_ribo_mask_keeps_the_s6_kinases():
    """Reading the Sanger flags beats matching on the symbol, and this is why.

    A real IBDverse load keeps 18 symbols that a ``^RP[LS]``/``^MRP[SL]`` regex
    would have removed: the ribosomal protein S6 kinases (``RPS6KA1``-``6``,
    ``RPS6KB1``/``2``, ``RPS6KC1``, ``RPS6KL1``) are signalling kinases, and
    the rest are antisense, divergent, or readthrough transcripts
    (``RPL34-AS1``, ``MRPS30-DT``, ``RPL17-C18orf32``, ...). None carries a
    ``gene_group__*`` flag. Dropping them would be a silent loss of 18
    signalling genes, several of which are in the LR table's downstream space.
    """
    var = _var()
    mask = mito_ribo_mask(var)
    symbols = var["gene_symbols"]

    for kept in ("RPS6KA1", "RPS6KB2", "MRPS30-DT"):
        assert not mask[symbols == kept].any(), f"{kept} is not a ribosomal protein"
    # The genuinely ribosomal neighbours in the same fixture still go.
    assert mask[symbols == "RPL13"].all()
    assert mask[symbols == "MRPS12"].all()
