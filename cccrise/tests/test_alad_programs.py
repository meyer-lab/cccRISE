"""`run_alad_programs`' table builders on small inputs."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from cccrise.runners import run_alad_programs as R


def test_detected_fractions_count_positive_entries_per_group():
    class Cells:
        X = csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]))
        var_names = pd.Index(["A", "B"])

    labels = np.array(["sender", "receiver", "sender", "rest"], dtype=object)
    detected = R.detected_fractions(Cells(), labels)  # ty: ignore[invalid-argument-type]
    assert detected.loc["A", "sender"] == 1.0
    assert detected.loc["B", "receiver"] == 1.0
    assert detected.loc["A", "rest"] == 0.0


def test_permutation_table_keeps_each_top_pair_and_its_test():
    test = pd.DataFrame(
        {
            "Component": 4,
            "LR pair": ["a", "b", "c"],
            "Rank": [2, 1, 3],
            "Observed": [0.1, 0.2, 0.3],
            "P": [0.5, 0.01, 0.9],
        }
    )
    null = np.arange(6, dtype=float).reshape(2, 3)  # permutations x pairs
    table = R.permutation_table(test, null, np.array([1, 0]))
    assert len(table) == 4
    b = table[table["LR pair"] == "b"]
    assert b["Null difference"].tolist() == [1.0, 4.0]
    assert set(b["Rank"]) == {1} and set(b["P"]) == {0.01}
