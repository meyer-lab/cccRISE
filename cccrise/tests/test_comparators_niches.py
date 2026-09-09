"""Unit tests for cccrise.comparators.niches."""

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from cccrise.comparators.niches import niches


def _small_adata() -> anndata.AnnData:
    # 6 cells, 4 genes, 2 celltypes (3 cells each).
    gene_names = ["L1", "R1", "R2a", "R2b"]
    X = np.array(
        [
            [2.0, 0.0, 3.0, 4.0],  # ct_A
            [1.0, 5.0, 0.0, 0.0],  # ct_A
            [3.0, 2.0, 1.0, 1.0],  # ct_A
            [0.0, 4.0, 2.0, 2.0],  # ct_B
            [1.0, 1.0, 0.0, 3.0],  # ct_B
            [2.0, 0.0, 1.0, 1.0],  # ct_B
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {"celltype": ["ct_A", "ct_A", "ct_A", "ct_B", "ct_B", "ct_B"]},
        index=[f"cell_{i}" for i in range(6)],
    )
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


def test_niches_covers_all_celltype_combinations():
    adata = _small_adata()
    signal, meta = niches(adata, groupby="celltype", lr_pairs=_lr_pairs(), random_state=0)

    combos = set(zip(meta["sending_type"], meta["receiving_type"]))
    assert combos == {
        ("ct_A", "ct_A"),
        ("ct_A", "ct_B"),
        ("ct_B", "ct_A"),
        ("ct_B", "ct_B"),
    }
    assert list(signal.index) == ["L1^R1", "L1^R2a&R2b"]
    assert list(signal.columns) == list(meta.index)


def test_niches_is_reproducible_with_fixed_seed():
    adata = _small_adata()
    signal1, meta1 = niches(adata, groupby="celltype", lr_pairs=_lr_pairs(), random_state=0)
    signal2, meta2 = niches(adata, groupby="celltype", lr_pairs=_lr_pairs(), random_state=0)

    pd.testing.assert_frame_equal(signal1, signal2)
    pd.testing.assert_frame_equal(meta1, meta2)


def test_niches_signal_is_elementwise_ligand_receptor_product():
    adata = _small_adata()
    signal, meta = niches(adata, groupby="celltype", lr_pairs=_lr_pairs(), random_state=1)

    for edge_id in signal.columns:
        row = meta.loc[edge_id]
        lig = adata[row["sender_cell"], "L1"].X.toarray().item()
        rec = adata[row["receiver_cell"], "R1"].X.toarray().item()
        np.testing.assert_allclose(signal.loc["L1^R1", edge_id], lig * rec)
