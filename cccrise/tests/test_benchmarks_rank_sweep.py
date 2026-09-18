"""Unit tests for cccrise.benchmarks.rank_sweep."""

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

from cccrise.benchmarks.rank_sweep import (
    RankSweepTensor,
    pseudobulk_sweep_tensor,
)


def _adata(n_conditions: int = 4, cells_per: int = 5, n_genes: int = 6):
    rng = np.random.default_rng(0)
    n = n_conditions * cells_per
    X = rng.random((n, n_genes)) + 0.1
    obs = pd.DataFrame(
        {
            "sample": np.repeat([f"s{i}" for i in range(n_conditions)], cells_per),
            "celltype": np.array(["A", "B"])[np.arange(n) % 2],
            "condition_unique_idxs": np.repeat(np.arange(n_conditions), cells_per),
        },
        index=[f"c{i}" for i in range(n)],
    )
    # Uppercase, as real human gene symbols are: build_context_ccc_tensor
    # uppercases the LR names under upper_letter_comparison but not the
    # expression index, so lowercase names would never match.
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    ad = anndata.AnnData(X=csr_array(X), obs=obs, var=var)
    ad.var["means"] = np.asarray(X.mean(axis=0))
    return ad


def _sweep(shape=(4, 5, 5, 7)) -> RankSweepTensor:
    rng = np.random.default_rng(1)
    return RankSweepTensor(
        key="fake",
        label="Fake",
        tensor=rng.random(shape),
        adata=_adata(n_conditions=shape[0]),
        condition_column="sample",
        cp_n_iter_max=50,
        cp_tol=1e-6,
        standardize=False,
        random_state=0,
        extra={},
    )


def test_max_rank_is_the_smallest_mode():
    # CP rank cannot exceed the smallest mode; for the pseudobulk tensor that
    # mode is the cell-type count, which is the structural cap the figure marks.
    assert _sweep((4, 5, 5, 7)).max_rank == 4
    assert _sweep((47, 8, 8, 513)).max_rank == 8
    assert _sweep((47, 19, 19, 513)).max_rank == 19


def test_usable_ranks_drops_ranks_above_the_cap():
    sweep = _sweep((10, 3, 3, 9))
    assert sweep.usable_ranks([1, 2, 3, 4, 5]) == [1, 2, 3]


def test_scores_at_returns_one_column_per_component():
    sweep = _sweep((4, 5, 5, 7))
    for rank in (1, 2, 3):
        scored = sweep.scores_at(rank)
        assert scored.scores.shape == (4, rank)
        assert scored.rank == rank
        assert list(scored.condition_names) == ["s0", "s1", "s2", "s3"]
        assert np.isfinite(scored.scores).all()


def test_scores_above_the_cap_are_refused():
    sweep = _sweep((4, 5, 5, 7))
    with pytest.raises(ValueError):
        sweep.scores_at(5)


def test_iter_scores_yields_only_usable_ranks():
    sweep = _sweep((4, 5, 5, 7))
    assert [rank for rank, _ in sweep.iter_scores([1, 2, 9])] == [1, 2]


def test_is_deterministic_for_a_fixed_seed():
    a = _sweep().scores_at(2)
    b = _sweep().scores_at(2)
    np.testing.assert_allclose(a.scores, b.scores)


def test_pseudobulk_builder_caps_at_the_cell_type_count():
    # Two cell types in, so the sender/receiver modes are length 2 and the
    # tensor cannot carry more than 2 CP components -- the whole reason the
    # pseudobulk curve stops short of CCC-RISE's.
    adata = _adata(n_conditions=3, cells_per=4, n_genes=4)
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["G0", "G1"],
            "receptor": ["G2", "G3"],
            "interaction_symbol": ["G0^G2", "G1^G3"],
        }
    )
    sweep = pseudobulk_sweep_tensor(
        adata, lr_pairs, condition_column="sample", groupby="celltype", complex_sep=None
    )
    assert sweep.tensor.shape[1] == sweep.tensor.shape[2] == 2
    assert sweep.max_rank == 2
    assert sweep.extra["n_cell_types"] == 2
    assert sweep.standardize is False
    # And its CP settings mirror run_pseudobulk_workflow's, so a point on the
    # curve is the same object that workflow produces.
    assert (sweep.cp_n_iter_max, sweep.cp_tol) == (1000, 1e-11)
