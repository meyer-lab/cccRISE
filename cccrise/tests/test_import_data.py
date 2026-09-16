"""Unit tests for cccrise.import_data."""

from typing import cast

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from cccrise.gene_filters import mito_ribo_mask
from cccrise.import_data import (
    IBD_NORMALIZATION,
    add_cond_idxs,
    import_ibdverse,
    import_ligand_receptor_pairs,
    prepare_dataset,
    use_gene_symbols,
)


def test_add_cond_idxs():
    n_cells = 30
    conditions = ["cond_B", "cond_A", "cond_C"]
    obs = pd.DataFrame(
        {"sample": [conditions[i % len(conditions)] for i in range(n_cells)]},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=["gene1", "gene2"])
    X = np.ones((n_cells, 2), dtype=np.float32)
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    adata = add_cond_idxs(adata, "sample")
    assert "condition_unique_idxs" in adata.obs.columns

    unique_idxs = adata.obs["condition_unique_idxs"].to_numpy()
    assert unique_idxs.min() == 0
    assert unique_idxs.max() == 2
    assert len(np.unique(unique_idxs)) == 3

    # All cells of the same sample must share the exact same condition index
    for cond in conditions:
        mask = adata.obs["sample"] == cond
        cond_idx_set = set(unique_idxs[mask])
        assert len(cond_idx_set) == 1


def test_prepare_dataset_filtering_and_normalization():
    # Construct 10 cells x 6 genes sparse count matrix
    # Cell 0: all zeros (total counts <= 10, should be filtered out)
    # Cells 1-9: nonzero counts > 10
    # Gene 0: zero across all cells (mean = 0 <= geneThreshold, filtered out)
    # Genes 1-5: expression above threshold
    n_cells, n_genes = 10, 6
    counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in range(1, n_cells):
        counts[i, 1:] = np.arange(1, n_genes) * (i + 1)

    obs = pd.DataFrame(
        {"patient": [f"P{i % 2}" for i in range(n_cells)]},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])
    adata = anndata.AnnData(X=csr_array(counts), obs=obs, var=var)

    processed = prepare_dataset(
        adata, condition_name="patient", geneThreshold=0.05, normalize=True
    )

    # Cell 0 filtered (9 remaining cells)
    assert processed.n_obs == 9
    # Gene 0 filtered (5 remaining genes)
    assert processed.n_vars == 5
    assert "gene_0" not in processed.var_names

    # Check that condition_unique_idxs and means exist
    assert "condition_unique_idxs" in processed.obs.columns
    assert "means" in processed.var.columns
    assert len(processed.var["means"]) == 5
    assert np.all(processed.var["means"] > 0)


def test_import_ligand_receptor_pairs_structure():
    df = import_ligand_receptor_pairs()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "ligand" in df.columns
    assert "receptor" in df.columns
    assert "interaction_symbol" in df.columns

    # Verify formatting of complex components (standardized uppercase)
    ligands = df["ligand"].astype(str)
    receptors = df["receptor"].astype(str)
    assert not ligands.str.contains(" ").any()
    assert not receptors.str.contains(" ").any()


def test_import_balf_covid():
    from cccrise.import_data import import_balf_covid

    adata = import_balf_covid(gene_threshold=0.001, normalize=True)
    assert isinstance(adata, anndata.AnnData)
    assert adata.n_obs > 0
    assert adata.n_vars > 0
    assert "condition_unique_idxs" in adata.obs.columns
    assert "means" in adata.var.columns


def test_import_alad():
    import os

    from cccrise.import_data import import_alad

    if os.path.exists("/opt/IVCSC/BAL-scRNAseq.raw.h5ad"):
        adata = import_alad(gene_threshold=0.001, normalize=True)
        assert isinstance(adata, anndata.AnnData)
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert "condition_unique_idxs" in adata.obs.columns
        assert "celltype" in adata.obs.columns
        assert "means" in adata.var.columns


def test_import_alad_brush():
    import os

    from cccrise.import_data import import_alad_brush

    if os.path.exists("/opt/IVCSC/alad.brush.sc.h5ad"):
        adata = import_alad_brush(gene_threshold=0.01, normalize=True)
        assert isinstance(adata, anndata.AnnData)
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert "condition_unique_idxs" in adata.obs.columns
        assert "means" in adata.var.columns


def _ensembl_indexed_adata() -> anndata.AnnData:
    """An AnnData shaped like IBDverse: Ensembl var_names, symbols in a column."""
    return anndata.AnnData(
        X=np.arange(12, dtype=float).reshape(3, 4),
        var=pd.DataFrame(
            # Two Ensembl IDs map to the same symbol, and one has no symbol.
            {"gene_symbols": ["TGFB1", "ACVR1", "TGFB1", "nan"]},
            index=["ENSG01", "ENSG02", "ENSG03", "ENSG04"],
        ),
    )


def test_use_gene_symbols_reindexes_and_keeps_the_first_symbol_matchable():
    """The LR table is keyed by symbol, so the bare symbol must survive."""
    adata = use_gene_symbols(_ensembl_indexed_adata())

    names = list(adata.var_names)
    # First occurrence keeps the bare symbol; the duplicate is suffixed.
    assert names[0] == "TGFB1"
    assert names[1] == "ACVR1"
    assert names[2].startswith("TGFB1") and names[2] != "TGFB1"
    # A gene with no symbol falls back to its own identifier.
    assert names[3] == "ENSG04"
    # The original identifiers stay recoverable.
    assert list(adata.var["ensembl_id"]) == ["ENSG01", "ENSG02", "ENSG03", "ENSG04"]


def test_use_gene_symbols_is_a_noop_without_the_column():
    """Datasets already keyed by symbol, like ALAD, must pass through untouched."""
    adata = anndata.AnnData(
        X=np.zeros((2, 2)), var=pd.DataFrame(index=["TGFB1", "ACVR1"])
    )
    result = use_gene_symbols(adata)
    assert list(result.var_names) == ["TGFB1", "ACVR1"]
    assert "ensembl_id" not in result.var


def _ibdverse_like_counts(tmp_path) -> str:
    """A packed `.h5ad` shaped like IBDverse: Sanger gene flags, one mito gene
    carrying a large share of every cell's depth."""
    import scipy.sparse as sp
    import vsparse

    rng = np.random.default_rng(0)
    n_cells, n_genes = 200, 60
    x = np.ceil(
        sp.random_array((n_cells, n_genes), density=0.3, rng=rng).toarray() * 20
    )
    x[:, 1] = 400.0  # the mito gene: most of each cell's counts

    symbols = [f"G{i}" for i in range(n_genes)]
    symbols[1] = "MT-CO1"
    symbols[2] = "RPL13"
    adata = anndata.AnnData(
        X=csr_array(x),
        obs=pd.DataFrame(
            {"sanger_sample_id": np.resize(["s1", "s2"], n_cells)},
            index=[f"c{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(
            {
                "gene_symbols": symbols,
                "gene_group__mito_transcript": [s == "MT-CO1" for s in symbols],
                "gene_group__ribo_protein": [s == "RPL13" for s in symbols],
            },
            index=[f"ENSG{i:05d}" for i in range(n_genes)],
        ),
    )
    path = str(tmp_path / "ibdlike.h5ad")
    vsparse.VCSCAnnData.from_anndata(adata, format="csr").write_h5ad(
        path, format="ivcsc"
    )
    return path


def test_ibdverse_excludes_genes_before_measuring_cell_depth(tmp_path):
    """Dropping the mito/ribo genes after normalizing is a different matrix.

    Those genes carry much of each cell's total, so they set the depth divisor
    every other gene is scaled by. If a refactor ever moves the exclusion after
    `normalized()` -- the obvious-looking simplification, since it reads as just
    dropping columns -- every surviving value changes and nothing raises. This
    is the only thing that would notice.
    """
    import vsparse

    path = _ibdverse_like_counts(tmp_path)
    loaded = import_ibdverse(filename=path, gene_threshold=0.0)

    # The same genes removed from an already-normalized matrix instead.
    packed = vsparse.load_packed(path)
    post_hoc_full = packed.normalized(IBD_NORMALIZATION).to_scipy_sparse(
        dtype=np.float32
    )
    keep = ~mito_ribo_mask(cast(pd.DataFrame, packed.var))
    post_hoc = post_hoc_full[:, keep]

    # The fixture is built so nothing is dropped by the count thresholds, which
    # makes the two matrices cell-for-cell and gene-for-gene comparable -- the
    # difference below is then the divisor alone, not a misalignment.
    assert loaded.shape == post_hoc.shape
    filtered_first = cast(csr_array, loaded.X).toarray()
    assert not np.allclose(filtered_first, post_hoc.toarray(), atol=1e-6)


def test_mito_ribo_mask_keeps_the_s6_kinases():
    """Reading the Sanger flags beats matching on the symbol, and this is why.

    A real IBDverse load keeps 18 symbols a `^RP[LS]`/`^MRP[SL]` regex would
    remove: the ribosomal protein S6 kinases (`RPS6KA1`-`6`, `RPS6KB1`/`2`,
    `RPS6KC1`, `RPS6KL1`) are signalling kinases, and the rest are antisense,
    divergent or readthrough transcripts (`RPL34-AS1`, `MRPS30-DT`,
    `RPL17-C18orf32`, ...). None carries a `gene_group__*` flag. Dropping them
    would silently lose 18 signalling genes, several downstream of pairs in the
    ligand-receptor table.
    """
    symbols = ["RPS6KA1", "RPS6KB2", "MRPS30-DT", "RPL34-AS1", "RPL13", "MRPS12"]
    ribosomal = {"RPL13", "MRPS12"}
    var = pd.DataFrame(
        {
            "gene_symbols": symbols,
            "gene_group__mito_transcript": [False] * len(symbols),
            # Flags as the Sanger pipeline sets them: on the ribosomal proteins,
            # not on everything whose symbol starts with RP.
            "gene_group__ribo_protein": [s in ribosomal for s in symbols],
        },
        index=[f"ENSG{i:05d}" for i in range(len(symbols))],
    )

    mask = mito_ribo_mask(var)
    kept = set(np.asarray(symbols)[~mask])
    assert {"RPS6KA1", "RPS6KB2", "MRPS30-DT", "RPL34-AS1"} <= kept
    assert set(np.asarray(symbols)[mask]) == ribosomal
