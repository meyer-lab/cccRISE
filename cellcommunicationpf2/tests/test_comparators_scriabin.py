"""Unit tests for cellcommunicationpf2.comparators.scriabin."""

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from cellcommunicationpf2.comparators.scriabin import scriabin


def _small_adata() -> anndata.AnnData:
    # 4 cells, 4 genes: a simple ligand/receptor pair plus a 2-subunit
    # receptor complex, so both plain and complex scoring paths get hit.
    gene_names = ["L1", "R1", "R2a", "R2b"]
    X = np.array(
        [
            [2.0, 0.0, 3.0, 4.0],  # cell_0
            [0.0, 5.0, 0.0, 0.0],  # cell_1
            [1.0, 1.0, 2.0, 2.0],  # cell_2
            [0.0, 0.0, 0.0, 0.0],  # cell_3 (silent)
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(4)])
    var = pd.DataFrame(index=gene_names)
    return anndata.AnnData(X=csr_array(X), obs=obs, var=var)


def _lr_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand": ["L1", "L1"],
            "receptor": ["R1", "R2a&R2b"],
            "interaction_symbol": ["L1^R1", "L1^R2a&R2b"],
        }
    )


def test_scriabin_returns_expected_columns_and_positive_edges():
    adata = _small_adata()
    result = scriabin(
        adata,
        send_cells=["cell_0", "cell_1", "cell_2"],
        rec_cells=["cell_0", "cell_1", "cell_2", "cell_3"],
        lr_pairs=_lr_pairs(),
    )

    expected_cols = {
        "sender",
        "receiver",
        "lr_pair",
        "ligand",
        "receptor",
        "ligand_expression",
        "receptor_expression",
        "edgeweight",
    }
    assert expected_cols == set(result.columns)
    assert (result["edgeweight"] > 0).all()
    # cell_3 is silent everywhere, so it can never be a receiver edge.
    assert "cell_3" not in result["receiver"].to_numpy()


def test_scriabin_complex_uses_min_aggregation():
    adata = _small_adata()
    result = scriabin(
        adata,
        send_cells=["cell_0"],
        rec_cells=["cell_0"],
        lr_pairs=_lr_pairs(),
        complex_agg_method="min",
    )

    complex_row = result[result["lr_pair"] == "L1^R2a&R2b"]
    assert len(complex_row) == 1
    # receptor expression for cell_0 is min(R2a=3, R2b=4) = 3, log1p'd.
    np.testing.assert_allclose(
        complex_row["receptor_expression"].iloc[0], np.log1p(3.0)
    )


def test_scriabin_exact_edgeweight_value():
    adata = _small_adata()
    result = scriabin(
        adata,
        send_cells=["cell_0"],
        rec_cells=["cell_1"],
        lr_pairs=_lr_pairs(),
    )
    simple_row = result[result["lr_pair"] == "L1^R1"]
    assert len(simple_row) == 1
    expected = np.log1p(2.0) * np.log1p(5.0)
    np.testing.assert_allclose(simple_row["edgeweight"].iloc[0], expected)
