"""Unit tests for cccrise.validation.secact (no network access)."""

import anndata
import numpy as np
import pandas as pd
import pytest

from cccrise.validation.secact import (
    group_signatures,
    pseudobulk_log_tpm_diff,
    ridge_activity,
    ridge_ligand_scores,
    score_rows,
    top_targets_long,
)


def _signatures(n_genes=300, n_sigs=6, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(n_genes)]
    return pd.DataFrame(
        rng.normal(size=(n_genes, n_sigs)),
        index=genes,
        columns=[f"L{j}" for j in range(n_sigs)],
    )


def test_ridge_activity_recovers_planted_signature():
    X = _signatures()
    rng = np.random.default_rng(1)
    Y = pd.DataFrame({"up": X["L2"] + 0.5 * rng.normal(size=len(X)), "down": -X["L4"]})
    z = ridge_activity(X, Y, lam=1.0, nrand=200)
    assert z["up"].idxmax() == "L2" and z.loc["L2", "up"] > 5
    assert z["down"].idxmin() == "L4" and z.loc["L4", "down"] < -5


def test_ridge_activity_uses_shared_genes_only():
    X = _signatures()
    Y = X[["L1"]].iloc[::2].rename(columns={"L1": "y"})
    z = ridge_activity(X, Y, lam=1.0, nrand=100)
    assert z["y"].idxmax() == "L1"


def test_group_signatures_merges_near_duplicates_and_drops_constant():
    X = _signatures()
    X["L0_copy"] = X["L0"] * 2 + 1e-3
    X["flat"] = 0.0
    grouped, groups = group_signatures(X, group_cor=0.9)
    assert groups["L0"] == groups["L0_copy"]
    assert "flat" not in groups.index
    assert grouped.shape[1] == X.shape[1] - 2


def test_ridge_ligand_scores_grouped_ligands_share_score():
    X = _signatures()
    X["L0_copy"] = X["L0"]
    grouped, groups = group_signatures(X)
    z = ridge_ligand_scores(X["L0"], grouped, groups, lam=1.0, nrand=100)
    assert z["L0"] == z["L0_copy"] and z.idxmax() in {"L0", "L0_copy"}


def test_pseudobulk_log_tpm_diff():
    counts = anndata.AnnData(
        np.array([[10, 0, 0], [10, 0, 0], [0, 10, 0], [0, 10, 0]], dtype=float),
        var=pd.DataFrame(index=["a", "b", "c"]),
    )
    diff = pseudobulk_log_tpm_diff(counts, np.array([True, True, False, False]))
    assert list(diff.index) == ["a", "b"]  # "c" undetected in both groups
    assert diff["a"] == pytest.approx(np.log2(1e6 + 1))
    assert diff["b"] == pytest.approx(-np.log2(1e6 + 1))


def test_top_targets_long_keeps_highest_weights():
    X = _signatures(n_genes=20, n_sigs=2)
    long = top_targets_long(X, top_n=3)
    assert len(long) == 6
    assert set(long.loc[long.ligand == "L0", "target"]) == set(
        X["L0"].nlargest(3).index
    )


def test_score_rows_universe_restricts_top_set():
    lr_pairs = pd.DataFrame({"ligand": ["A", "B", "C"], "receptor": ["r"] * 3})
    D = np.array([[1.0], [0.9], [0.1]])
    scores = {"A": 0.7, "B": 0.6, "C": 0.5}
    own = pd.DataFrame(score_rows(1, scores, D, lr_pairs)).set_index("ligand")
    assert own["is_top"].to_dict() == {"A": True, "B": True, "C": False}
    shared = pd.DataFrame(score_rows(1, scores, D, lr_pairs, universe={"B", "C"}))
    shared = shared.set_index("ligand")
    assert shared["is_top"].to_dict() == {"A": False, "B": True, "C": False}
    assert np.isnan(shared.loc["A", "score"])
    assert shared.loc["B", "rel_weight"] == pytest.approx(1.0)
