"""Unit tests for cccrise.gene_filters and cccrise._vsparse_filter."""

from typing import cast

import anndata
import numpy as np
import pandas as pd
import pytest
import vsparse
from scipy.sparse import csr_array

from cccrise._vsparse_filter import load_and_normalize_excluding
from cccrise.gene_filters import mito_ribo_mask


def _dense(adata):
    """`adata.X` as a dense array, whatever sparse container it arrived in."""
    return np.asarray(adata.X.todense())  # ty: ignore[unresolved-attribute]


def _var(symbols, *, mito=(), ribo=()):
    """A `var` table shaped like IBDverse's, with the Sanger group flags."""
    symbols = list(symbols)
    return pd.DataFrame(
        {
            "gene_symbols": symbols,
            "gene_group__mito_transcript": [s in mito for s in symbols],
            "gene_group__mito_protein": [s in mito for s in symbols],
            "gene_group__ribo_protein": [s in ribo for s in symbols],
            "gene_group__ribo_rna": [False] * len(symbols),
        },
        index=[f"ENSG{i:011d}" for i in range(len(symbols))],
    )


def _excluded(var, **kwargs):
    symbols = var["gene_symbols"] if "gene_symbols" in var else pd.Series(var.index)
    return set(np.asarray(symbols)[mito_ribo_mask(var, **kwargs)])


def test_flagged_mito_and_cytosolic_ribosomal_are_excluded():
    var = _var(
        ["MT-CO1", "MT-ND3", "RPS18", "RPL10", "FAU", "ACTB", "IL7R"],
        mito=["MT-CO1", "MT-ND3"],
        ribo=["RPS18", "RPL10", "FAU"],
    )
    assert _excluded(var) == {"MT-CO1", "MT-ND3", "RPS18", "RPL10", "FAU"}


def test_cytosolic_ribosomal_proteins_the_shipped_flag_misses():
    """RPL13/RPL18/RPSA are real ribosomal proteins that IBDverse leaves unflagged."""
    var = _var(["RPL13", "RPL18", "RPSA", "ACTB"], ribo=[])
    assert _excluded(var) == {"RPL13", "RPL18", "RPSA"}


def test_ribosomal_protein_s6_kinases_are_kept():
    """A `^RP[SL]` regex would sweep these up; they are signaling genes."""
    var = _var(["RPS6KA3", "RPS6KB1", "RPS6KC1", "RPS6KL1", "RPL34-AS1", "RPS10-NUDT3"])
    assert _excluded(var) == set()


def test_mito_ribosomal_proteins_are_kept_by_default_and_optional():
    """`gene_group__ribo_protein` conflates the two ribosomes; we split them."""
    var = _var(
        ["MRPL20", "MRPS21", "DAP3", "RPS18"],
        ribo=["MRPL20", "MRPS21", "DAP3", "RPS18"],
    )
    assert _excluded(var) == {"RPS18"}
    assert _excluded(var, exclude_mito_ribosomal=True) == {
        "MRPL20",
        "MRPS21",
        "DAP3",
        "RPS18",
    }


def test_mtrnr2l_pseudogenes_are_excluded_by_default_and_optional():
    var = _var(["MTRNR2L12", "MTRNR2L8", "MTRNR2L1", "ACTB"])
    assert _excluded(var) == {"MTRNR2L12", "MTRNR2L8", "MTRNR2L1"}
    assert _excluded(var, exclude_mtrnr2l=False) == set()


def test_unannotated_var_is_passed_through_untouched():
    """A cohort without the Sanger flags must not be half-filtered silently."""
    var = pd.DataFrame(index=["ACTB", "IL7R", "CD4"])
    assert not mito_ribo_mask(var).any()


def test_symbols_fall_back_to_the_index():
    var = pd.DataFrame(index=["MTRNR2L12", "ACTB"])
    assert _excluded(var) == {"MTRNR2L12"}


# -- load_and_normalize_excluding -------------------------------------------


@pytest.fixture
def ivcsc_file(tmp_path):
    """A small IVCSR-backed h5ad with two genes worth excluding."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 8
    counts = rng.poisson(4.0, size=(n_cells, n_genes)).astype(np.float32)
    # Make the excluded genes dominate the library, so their removal visibly
    # rescales everything else.
    counts[:, 0] *= 40  # MT-CO1
    counts[:, 1] *= 20  # RPS18
    counts[counts.sum(axis=1) == 0, 2] = 1.0

    symbols = ["MT-CO1", "RPS18", "ACTB", "IL7R", "CD4", "CD8A", "CCL5", "CXCL13"]
    var = _var(symbols, mito=["MT-CO1"], ribo=["RPS18"])
    obs = pd.DataFrame(
        {"sample": [f"s{i % 3}" for i in range(n_cells)]},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    adata = anndata.AnnData(X=csr_array(counts), obs=obs, var=var)

    path = tmp_path / "small.ivcsc.h5ad"
    vsparse.VCSCAnnData(
        vsparse.from_anndata(adata, format="csr"), obs=obs, var=var
    ).write_h5ad(path, format="ivcsc")
    return path


def test_no_exclusion_reproduces_vsparse_exactly(ivcsc_file):
    """The mirrored implementation must not drift from upstream."""
    reference = vsparse.load_and_normalize(ivcsc_file, min_cell_counts=10.0)
    ours = load_and_normalize_excluding(ivcsc_file, min_cell_counts=10.0)

    assert ours.shape == reference.shape
    assert list(ours.var_names) == list(reference.var_names)
    assert list(ours.obs_names) == list(reference.obs_names)
    np.testing.assert_array_equal(_dense(ours), _dense(reference))


def test_exclusion_drops_the_genes_and_rescales_the_rest(ivcsc_file):
    """The point of filtering at load time: retained genes must change too.

    If the excluded genes were dropped from the returned object instead, the
    per-cell depth divisor would still include them and every retained gene
    would be bit-identical to the unfiltered run.
    """
    unfiltered = vsparse.load_and_normalize(ivcsc_file, min_cell_counts=10.0)
    filtered = load_and_normalize_excluding(
        ivcsc_file, min_cell_counts=10.0, exclude_genes=mito_ribo_mask
    )

    assert set(filtered.var["gene_symbols"]) == {
        "ACTB",
        "IL7R",
        "CD4",
        "CD8A",
        "CCL5",
        "CXCL13",
    }

    kept = list(filtered.var_names)
    assert not np.allclose(_dense(unfiltered[:, kept]), _dense(filtered))


def test_post_hoc_column_drop_is_not_equivalent(ivcsc_file):
    """Pin the failure mode the module exists to avoid."""
    unfiltered = vsparse.load_and_normalize(ivcsc_file, min_cell_counts=10.0)
    mask = mito_ribo_mask(cast(pd.DataFrame, unfiltered.var))
    post_hoc = unfiltered[:, ~mask]
    filtered = load_and_normalize_excluding(
        ivcsc_file, min_cell_counts=10.0, exclude_genes=mito_ribo_mask
    )

    assert list(post_hoc.var_names) == list(filtered.var_names)
    assert not np.allclose(_dense(post_hoc), _dense(filtered))


def test_exclusion_is_validated(ivcsc_file):
    with pytest.raises(ValueError, match="one-dimensional mask"):
        load_and_normalize_excluding(
            ivcsc_file, exclude_genes=lambda var: np.zeros(3, dtype=bool)
        )
    with pytest.raises(ValueError, match="boolean mask"):
        load_and_normalize_excluding(
            ivcsc_file, exclude_genes=lambda var: np.zeros(len(var), dtype=int)
        )
    with pytest.raises(ValueError, match="removed every gene"):
        load_and_normalize_excluding(
            ivcsc_file, exclude_genes=lambda var: np.ones(len(var), dtype=bool)
        )


def test_mask_is_applied_before_symbol_reindexing(ivcsc_file):
    """The filter reads the file's own `var`, so it runs on Ensembl-keyed input.

    `use_gene_symbols` re-indexes afterwards; the two must not fight -- the
    excluded genes stay gone and the survivors come back keyed by symbol.
    """
    from cccrise.import_data import use_gene_symbols

    filtered = load_and_normalize_excluding(
        ivcsc_file, min_cell_counts=10.0, exclude_genes=mito_ribo_mask
    )
    named = use_gene_symbols(filtered)

    assert set(named.var_names) == {"ACTB", "IL7R", "CD4", "CD8A", "CCL5", "CXCL13"}
    assert "MT-CO1" not in set(named.var_names)
    assert "RPS18" not in set(named.var_names)
