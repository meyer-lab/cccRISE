"""Unit tests for cccrise.comparators.inputs."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

from cccrise.ccc_rise import add_complexes_to_expression
from cccrise.comparators.inputs import (
    Inputs,
    entity_expression,
    unique_lr_pairs,
    with_complex_pseudogenes,
)

GENES = ["A", "B", "C", "D"]
X = np.array([[1.0, 2.0, 0.5, 4.0], [3.0, 0.2, 2.0, 1.0], [0.5, 5.0, 1.5, 2.5]])
ENTITIES = ["A", "A&B", "B&C&D", "A&E"]


@pytest.mark.parametrize("agg", ["min", "mean", "gmean"])
def test_matches_add_complexes_to_expression(agg):
    """Same values cccRISE's own complex aggregation gives (genes x cells)."""
    complexes = {e: e.split("&") for e in ENTITIES if "&" in e}
    ref = add_complexes_to_expression(
        pd.DataFrame(X.T, index=GENES), complexes, agg_method=agg
    )
    ours = entity_expression(X, GENES, ENTITIES, agg)
    np.testing.assert_allclose(ours, ref.loc[ENTITIES].to_numpy().T)


def test_sparse_input_and_product():
    ours = entity_expression(csr_array(X), GENES, ENTITIES, "product")
    expected = np.column_stack(
        [X[:, 0], X[:, 0] * X[:, 1], X[:, 1] * X[:, 2] * X[:, 3], np.zeros(3)]
    )
    np.testing.assert_allclose(ours, expected)


def test_unknown_aggregator_raises():
    with pytest.raises(ValueError):
        entity_expression(X, GENES, ["A&B"], "max")


def test_pseudogenes_append_complexes_once():
    lr = pd.DataFrame({"ligand": ["A", "A&B"], "receptor": ["B&C&D", "A&B"]})
    combined, names = with_complex_pseudogenes(csr_array(X), GENES, lr)
    assert list(names) == [*GENES, "A&B", "B&C&D"]
    np.testing.assert_allclose(combined.toarray()[:, 4], np.minimum(X[:, 0], X[:, 1]))


def test_inputs_rows_and_per_sample_expression():
    obs = pd.DataFrame({"sample": ["s2", "s1", "s2"], "celltype": ["t", "t", "u"]})
    lr = pd.DataFrame({"ligand": ["A"], "receptor": ["B&C&D"]})
    inp = Inputs(X, pd.Index(GENES), obs, lr)
    assert inp.samples == ["s1", "s2"]
    rows = inp.rows("s2")
    assert rows.tolist() == [0, 2]
    lig, rec = inp.lr_expression(rows)
    np.testing.assert_allclose(lig[:, 0], X[rows, 0])
    np.testing.assert_allclose(rec[:, 0], X[rows, 1:].min(axis=1))


def test_unique_lr_pairs_keeps_first_of_same_genes():
    lr = pd.DataFrame(
        {
            "interaction_symbol": ["PLG^F2R", "A^B", "PLG^F2RL3"],
            "lr_name": ["PLG=F2RL3", "A=B", "PLG=F2RL3"],
        }
    )
    out = unique_lr_pairs(lr)
    assert out["interaction_symbol"].tolist() == ["PLG^F2R", "A^B"]
