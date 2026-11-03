"""Unit tests for cccrise.comparators.niches."""

import numpy as np
import pandas as pd
import pytest

from cccrise.comparators import niches

LIG = np.array([[2.0, 0.0], [1.0, 3.0], [0.0, 1.0]])  # 3 cells x 2 LR
REC = np.array([[1.0, 2.0], [0.0, 1.0], [4.0, 0.0]])


@pytest.mark.parametrize(
    "blend, system",
    [
        ("mean", LIG.mean(axis=0)),
        ("sum", LIG.sum(axis=0)),
        ("mean.adj", LIG.mean(axis=0) / (LIG > 0).mean(axis=0)),
    ],
)
def test_system_to_cell_blends(blend, system):
    np.testing.assert_allclose(niches.system_to_cell(LIG, REC, blend), system * REC)


def test_cell_to_system_is_the_mirror():
    np.testing.assert_allclose(
        niches.cell_to_system(LIG, REC, "mean"), LIG * REC.mean(axis=0)
    )


def test_mean_adj_is_nan_for_unexpressed_entity():
    lig = np.zeros((2, 1))
    assert np.isnan(niches.system_to_cell(lig, np.ones((2, 1)), "mean.adj")).all()


def test_unknown_blend_raises():
    with pytest.raises(ValueError):
        niches.system_to_cell(LIG, REC, "median")


def test_cell_to_cell_pairs_block_sizes_and_no_repeats():
    types = np.array(["a"] * 5 + ["b"] * 3)
    pairs = niches.cell_to_cell_pairs(types, np.random.default_rng(0))
    sizes = pairs.groupby(["sending_type", "receiving_type"]).size().to_dict()
    assert sizes == {("a", "a"): 5, ("a", "b"): 3, ("b", "a"): 3, ("b", "b"): 3}
    for _, block in pairs.groupby(["sending_type", "receiving_type"]):
        assert block["sender"].is_unique and block["receiver"].is_unique
    assert (types[pairs["sender"]] == pairs["sending_type"]).all()
    assert (types[pairs["receiver"]] == pairs["receiving_type"]).all()


def test_cell_to_cell_is_elementwise_product():
    pairs = pd.DataFrame({"sender": [0, 1], "receiver": [2, 0]})
    out = niches.cell_to_cell(LIG, REC, pairs)
    np.testing.assert_allclose(out, [LIG[0] * REC[2], LIG[1] * REC[0]])


def test_filters():
    lr = pd.DataFrame({"ligand": ["A", "B", "A"], "receptor": ["X", "X&Y", "Z"]})
    kept = niches.filter_ground_truth(lr, ["A", "B", "X", "Y"])
    assert kept["receptor"].tolist() == ["X", "X&Y"]
    mask = niches.filter_cells_per_type(np.array(["a", "a", "b"]), 1)
    assert mask.tolist() == [True, True, False]
    assert niches.filter_cells_per_type(np.array(["a"]), None).all()
