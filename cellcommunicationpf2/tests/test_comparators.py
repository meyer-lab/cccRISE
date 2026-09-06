"""Unit tests for cellcommunicationpf2.comparators."""

import numpy as np
import pandas as pd
import pytest

from cellcommunicationpf2.comparators import staccato


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
