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
    table = scores.containment(span, inside.iloc[::-1])
    assert table["Containment"].iloc[0] == pytest.approx(1)
    values = span.to_numpy()
    basis = np.eye(30)[:, [0]]
    residual = basis - values @ np.linalg.lstsq(values, basis, rcond=None)[0]
    orthogonal = _frame(residual)
    table = scores.containment(span, orthogonal)
    assert table["Containment"].iloc[0] == pytest.approx(0, abs=1e-12)


def test_cccrise_components_without_cells_keep_sample_and_lr():
    factors = xr.Dataset(
        {
            "cp_A": (("condition", "cp_component"), np.ones((3, 2))),
            "cp_D": (("lr_pair", "cp_component"), np.arange(8.0).reshape(4, 2)),
        },
        coords={
            "condition": ["a", "b", "c"],
            "lr_pair": [f"L{i}^R{i}" for i in range(4)],
            "cp_component": [1, 2],
        },
    )
    components = scores.cccrise_components(factors)
    assert set(components.modes) == {"sample", "lr"}
    assert list(components.modes["lr"].columns) == [1, 2]
    assert components.modes["lr"].loc["L3^R3", 2] == 7


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
