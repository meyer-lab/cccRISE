"""Unit tests for cccrise.comparators.staccato."""

import numpy as np
import pandas as pd
import pytest

from cccrise.comparators.staccato import staccato


def test_staccato_unsupervised_shapes_and_reconstruction():
    rng = np.random.default_rng(42)
    tensor = rng.normal(size=(6, 8, 4, 4))
    core_shape = (3, 4, 3, 3)

    result = staccato(tensor, core_shape=core_shape, random_state=0)

    assert result.W1.shape == (6, 3)
    assert result.W2.shape == (8, 4)
    assert result.W3.shape == (4, 3)
    assert result.W4.shape == (4, 3)
    assert result.G.shape == core_shape
    assert result.fitted.shape == tensor.shape
    assert result.coef_tensor.shape == tensor.shape
    assert np.isfinite(result.sigma2)
    assert np.isfinite(result.log_likelihood)
    assert result.sigma2 >= 0


def test_staccato_with_supervised_mode1_covariate():
    rng = np.random.default_rng(1)
    n_samples = 10
    tensor = rng.normal(size=(n_samples, 6, 4, 4))
    covariate = np.column_stack(
        [np.ones(n_samples), (rng.uniform(size=n_samples) > 0.5).astype(float)]
    )

    result = staccato(
        tensor, core_shape=(2, 3, 3, 3), covariate=covariate, random_state=0
    )

    # Mode-1 factor lives in covariate-column space, not sample space.
    assert result.W1.shape == (2, 2)
    assert result.fitted.shape == tensor.shape


def test_staccato_labels_factors_when_names_given():
    rng = np.random.default_rng(7)
    tensor = rng.normal(size=(5, 4, 3, 3))
    lr_names = [f"lr{i}" for i in range(4)]
    sender_names = [f"send{i}" for i in range(3)]
    receiver_names = [f"rec{i}" for i in range(3)]

    result = staccato(
        tensor,
        core_shape=(2, 2, 2, 2),
        lr_names=lr_names,
        sender_names=sender_names,
        receiver_names=receiver_names,
        random_state=0,
    )

    assert isinstance(result.W2, pd.DataFrame)
    assert list(result.W2.index) == lr_names
    assert isinstance(result.W3, pd.DataFrame)
    assert list(result.W3.index) == sender_names
    assert isinstance(result.W4, pd.DataFrame)
    assert list(result.W4.index) == receiver_names

    # W1 is left unlabeled -- upstream STACCato doesn't name it either
    # (its rows index covariate columns, not a biological entity).
    assert isinstance(result.W1, np.ndarray)


def test_staccato_rejects_rank_larger_than_tensor():
    tensor = np.zeros((4, 4, 4, 4))
    with pytest.raises(ValueError, match="exceeds tensor shape"):
        staccato(tensor, core_shape=(5, 2, 2, 2))


def test_staccato_rejects_wrong_dimensionality():
    tensor = np.zeros((4, 4, 4))
    with pytest.raises(ValueError, match="4-mode tensor"):
        staccato(tensor, core_shape=(2, 2, 2, 2))


def _low_rank(rng, shape, ranks, noise):
    import tensorly as tl

    core = rng.normal(size=ranks)
    factors = [np.linalg.qr(rng.normal(size=(n, r)))[0] for n, r in zip(shape, ranks)]
    return tl.tucker_to_tensor((core, factors)) + noise * rng.normal(size=shape)


@pytest.mark.parametrize("supervised", [False, True])
def test_coefficient_tucker_matches_the_full_tensor_fit(supervised):
    """The second Tucker fit, done on the core, equals `tucker` on the full
    coefficient tensor (factors and core, signs included)."""
    from tensorly.decomposition import tucker

    rng = np.random.default_rng(3)
    tensor = _low_rank(rng, (12, 10, 5, 5), (4, 4, 3, 3), noise=0.1)
    covariate = (
        np.column_stack([np.ones(12), rng.normal(size=(12, 4))])
        if supervised
        else None
    )
    core_shape = (3, 4, 3, 3)
    result = staccato(tensor, core_shape, covariate=covariate, random_state=0)
    G, factors = tucker(
        result.coef_tensor, rank=list(core_shape), init="svd", random_state=0
    )
    for mine, full in zip(
        (result.W1, result.W2, result.W3, result.W4), factors, strict=True
    ):
        np.testing.assert_allclose(mine, full, atol=1e-10)
    np.testing.assert_allclose(result.G, G, atol=1e-10)


def test_staccato_without_full_output_keeps_the_factors():
    rng = np.random.default_rng(5)
    tensor = _low_rank(rng, (8, 9, 4, 4), (3, 3, 2, 2), noise=0.05)
    full = staccato(tensor, (3, 3, 2, 2), random_state=0)
    lean = staccato(tensor, (3, 3, 2, 2), random_state=0, full_output=False)
    assert lean.fitted is None and lean.coef_tensor is None
    assert np.isnan(lean.sigma2)
    for name in ("W1", "W2", "W3", "W4", "G"):
        np.testing.assert_array_equal(getattr(lean, name), getattr(full, name))
