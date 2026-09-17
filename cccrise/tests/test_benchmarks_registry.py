"""Unit tests for cccrise.benchmarks.registry."""

import anndata
import numpy as np
import pandas as pd
import pytest

from cccrise.benchmarks.registry import (
    METHODS,
    SINGLE_CELL_AXIS,
    TENSOR_AXIS,
    BenchmarkMethod,
    MethodScores,
    resolve_methods,
)


def test_every_registered_method_is_well_formed():
    for key, method in METHODS.items():
        assert isinstance(method, BenchmarkMethod)
        # The dict key is the address used by --methods; a mismatch would make
        # the CLI choices and the CSV's Method column disagree.
        assert method.key == key
        assert method.axis in (TENSOR_AXIS, SINGLE_CELL_AXIS)
        assert callable(method.fit)
        assert method.label and method.notes


def test_resolve_methods_defaults_to_all_and_preserves_order():
    assert resolve_methods(None) == list(METHODS.values())
    assert resolve_methods([]) == list(METHODS.values())
    keys = ["pseudobulk_cp", "ccc_rise"]
    assert [m.key for m in resolve_methods(keys)] == keys


def test_resolve_methods_refuses_an_unknown_key():
    # Silently skipping would drop a method from the table without saying so.
    with pytest.raises(ValueError, match="Unknown method"):
        resolve_methods(["ccc_rise", "tensor_cell2cell"])


def test_method_scores_validates_its_shape_contract():
    ok = MethodScores(
        scores=np.zeros((4, 3)),
        condition_names=np.array(["a", "b", "c", "d"]),
        rank=3,
    )
    assert ok.scores.shape == (4, 3)

    with pytest.raises(ValueError, match="n_conditions, n_components"):
        MethodScores(scores=np.zeros(4), condition_names=np.array(["a"]), rank=1)

    with pytest.raises(ValueError, match="condition names"):
        MethodScores(
            scores=np.zeros((4, 3)), condition_names=np.array(["a", "b"]), rank=3
        )


def test_condition_order_matches_the_label_order_helper():
    # The registry's row order and `get_sample_to_group`'s index order have to
    # be the same, or every method's scores are silently permuted relative to
    # the phenotype. Both are np.unique over the condition column.
    from cccrise.benchmarks.registry import _condition_order
    from cccrise.figures.commonFuncs.plotGeneral import get_sample_to_group

    obs = pd.DataFrame(
        {
            "sample": ["s2", "s1", "s2", "s3", "s1"],
            "status": ["alad", "control", "alad", "control", "control"],
        }
    )
    adata = anndata.AnnData(X=np.zeros((5, 2)), obs=obs)

    order = _condition_order(adata, "sample")
    labels = get_sample_to_group(adata, "sample", "status")
    assert list(order) == list(labels.index) == ["s1", "s2", "s3"]
