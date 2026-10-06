import numpy as np
import pandas as pd
from scipy import sparse

from cccrise.validation.program_breadth import (
    abs_cv,
    detection_table,
    lr_prevalence,
    mode_statistics,
    participation_ratio,
    top_pathways,
    weighted_prevalence,
)


def _detection():
    # Two samples x two types; gene L only in type a, receptor subunits R1, R2.
    X = sparse.csr_array(
        np.array(
            [
                # sample s1, type a: L and R1 and R2 on
                [1, 1, 1],
                [1, 0, 0],
                # sample s1, type b: R1 only
                [0, 1, 0],
                [0, 1, 0],
                # sample s2, type a: nothing
                [0, 0, 0],
                # sample s2, type b: R1 and R2
                [0, 1, 1],
            ]
        )
    )
    samples = ["s1", "s1", "s1", "s1", "s2", "s2"]
    types = ["a", "a", "b", "b", "a", "b"]
    return detection_table(X, ["L", "R1", "R2"], samples, types)


def test_detection_table_is_fraction_of_cells_per_sample_and_type():
    det = _detection()
    assert det.loc[("s1", "a"), "L"] == 1.0
    assert det.loc[("s1", "a"), "R1"] == 0.5
    assert det.loc[("s2", "b"), "R2"] == 1.0


def test_lr_prevalence_counts_blocks_with_ligand_and_complex_receptor():
    det = _detection()
    prev = lr_prevalence(det, pd.Series(["L"]), pd.Series(["R1&R2"]), "&", 0.5)
    # s1: L in {a}, R1&R2 (min subunit) >= 0.5 only in a -> 1 of 4 blocks;
    # s2: no ligand -> 0 of 4.
    assert prev[0] == 1 / 8


def test_lr_prevalence_is_nan_free_for_missing_genes():
    det = _detection()
    prev = lr_prevalence(det, pd.Series(["NOPE"]), pd.Series(["R1"]))
    assert prev[0] == 0


def test_breadth_statistics():
    even, single = np.ones((4, 1)), np.array([[1.0], [0], [0], [0]])
    assert participation_ratio(even)[0] == 4 and participation_ratio(single)[0] == 1
    assert abs_cv(even)[0] == 0
    assert weighted_prevalence(single, np.array([0.3, 1, 1, 1]))[0] == 0.3


def test_top_pathways_and_mode_statistics():
    lr = np.array([[3.0, 0], [2, 0], [0, 1], [1, 2]])
    assert top_pathways(lr, ["P", "P", "Q", "R"], n_top=2) == ["P", "R, Q"]
    modes = {
        "condition": np.ones((5, 2)),
        "lr": lr,
        "sender": np.ones((3, 2)),
        "receiver": np.ones((3, 2)),
    }
    table = mode_statistics(modes, np.full(4, 0.5), ["P", "P", "Q", "R"])
    assert set(table["statistic"]) == {
        "sample_breadth",
        "sample_abs_cv",
        "lr_prevalence",
        "type_breadth",
    }
    assert (table.loc[table["mode"] == "lr", "value"] == 0.5).all()
    assert (table.loc[table["mode"] != "lr", "top_pathways"] == "").all()
