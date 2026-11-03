"""Unit tests for cccrise.comparators.scores."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cccrise.comparators import scores
from cccrise.comparators.tensors import PooledTensor


def _frame(values, index=None):
    values = np.asarray(values, float)
    index = index if index is not None else [f"r{i}" for i in range(len(values))]
    return pd.DataFrame(values, index=index, columns=range(1, values.shape[1] + 1))


def test_containment_of_a_column_in_its_own_span():
    rng = np.random.default_rng(0)
    span = _frame(rng.standard_normal((30, 3)))
    inside = _frame(span.to_numpy() @ [[1.0], [2.0], [0.5]])
    table = scores.containment(span, inside.iloc[::-1], 50, rng)
    assert table["Containment"].iloc[0] == pytest.approx(1)
    assert table["Contained"].iloc[0]
    values = span.to_numpy()
    basis = np.eye(30)[:, [0]]
    residual = basis - values @ np.linalg.lstsq(values, basis, rcond=None)[0]
    orthogonal = _frame(residual)
    table = scores.containment(span, orthogonal, 50, rng)
    assert table["Containment"].iloc[0] == pytest.approx(0, abs=1e-12)
    assert not table["Contained"].iloc[0]


def test_effective_number_counts_equal_entries():
    columns = _frame([[1, 3], [1, 0], [-1, 0], [1, 0]])
    np.testing.assert_allclose(scores.effective_number(columns), [4, 1])
    np.testing.assert_allclose(scores.top_share(columns), [0.25, 1])


def test_oriented_makes_the_largest_entry_positive():
    columns = _frame([[0.1, 2], [-3, -1]])
    np.testing.assert_allclose(scores.oriented(columns), [[-0.1, 2], [3, -1]])


def test_tucker_column_profiles_map_a_rank_one_core_slice():
    rng = np.random.default_rng(1)
    shapes = {"sample": (5, 2), "sender": (4, 2), "receiver": (4, 3), "lr": (6, 2)}
    factors = {m: rng.random(s) for m, s in shapes.items()}
    u = [np.array([0.6, 0.8]), np.array([1.0, 0.0]), np.array([0, 0.6, 0.8])]
    core = np.zeros((2, 2, 3, 2))
    core[..., 1] = np.einsum("i,j,k->ijk", *u)
    fit = xr.Dataset(
        {f"{m}_factor": ((m, f"{m}_component"), f) for m, f in factors.items()}
        | {"core": (tuple(f"{m}_component" for m in shapes), core)}
    )
    profiles = scores.tucker_column_profiles(fit, 2)
    for mode, vector in zip(("sample", "sender", "receiver"), u, strict=True):
        expected = factors[mode] @ vector
        cosine = profiles[mode] @ expected
        cosine /= np.linalg.norm(profiles[mode]) * np.linalg.norm(expected)
        assert abs(cosine) == pytest.approx(1)


def _column(**overrides):
    samples = [f"s{i}" for i in range(6)]
    column = {
        "EffectiveSamples": 3.0,
        "EffectiveLR": 2.0,
        "Ligand": "L",
        "Receptor": "R",
        "TopSender": "a",
        "TopReceiver": "b",
        "SampleProfile": pd.Series([1.0, 3, 2, 5, 4, 6], index=samples),
    }
    return column | overrides


def test_classify_takes_the_first_matching_class():
    fold = pd.DataFrame({"L": [5.0, 1], "R": [1.0, 1]}, index=["a", "b"])
    detected = pd.DataFrame({"L": [0.5, 0.5], "R": [0.5, 0.5]}, index=["a", "b"])
    samples = [f"s{i}" for i in range(6)]
    fractions = pd.DataFrame(
        {"a": [0.2, 0.1, 0.3, 0.1, 0.2, 0.1], "b": [6.0, 5, 4, 3, 2, 1]},
        index=samples,
    )
    flat = pd.DataFrame(
        {"a": [0.3, 0.1, 0.2, 0.2, 0.1, 0.3], "b": [0.2, 0.3, 0.1, 0.2, 0.3, 0.1]},
        samples,
    )

    def classed(**overrides):
        return scores.classify(_column(**overrides), fold, detected, flat)

    assert classed(EffectiveSamples=1.2) == "one sample"
    assert classed(Receptor="L") == "marker or homophilic"
    assert classed() == "candidate biology"
    assert classed(EffectiveLR=40.0) == "diffuse"
    fold.loc["a", "R"] = 5.0
    assert classed() == "marker or homophilic"
    fold.loc["a", "R"] = 1.0
    assert scores.classify(_column(), fold, detected, fractions) == "abundance readout"


def test_inner_losses_count_what_dropping_a_type_removes():
    values = np.ones((2, 2, 2, 1))
    mask = np.ones_like(values)
    mask[1, 1] = 0
    mask[1, :, 1] = 0
    masked = PooledTensor(
        values, mask, np.array(["s1", "s2"]), np.array(["a", "b"]), np.array(["p"])
    )
    table = scores.inner_losses(masked, ["a"]).set_index("Measure")
    assert table.loc["cell types", "Share"] == 0.5
    # observed: s1 has 4 entries, s2 only a->a; dropping b removes s1's 3
    assert table.loc["observed entries", "Dropped"] == 3
    assert table.loc["signal", "Share"] == pytest.approx(3 / 5)


def test_weight_share_of_a_group_of_cells():
    weights = pd.DataFrame({1: [1.0, -1.0, 2.0], 2: [0.0, 0.0, 3.0]})
    shares = scores.weight_share(weights, np.array([True, False, False]))
    np.testing.assert_allclose(shares, [0.25, 0.0])
