"""Unit tests for cccrise.comparators.niches_programs."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from cccrise.comparators.niches_programs import (
    SIDES,
    TAUS,
    Cohort,
    label_mean,
    pair_tokens,
    population_table,
    program_score,
    tau_table,
    token_expression,
)


def test_pair_tokens_keeps_complexes_whole():
    table = pair_tokens(np.array(["A^B&C", "D&E^F"]))
    assert table["ligand"].tolist() == ["A", "D&E"]
    assert table["receptor"].tolist() == ["B&C", "F"]


def test_token_expression_takes_a_complexs_minimum_subunit():
    X = csr_matrix(np.array([[1.0, 2.0, 0.0], [3.0, 1.0, 5.0]]))
    genes = pd.Index(["A", "B", "C"])
    expression = token_expression(X, genes, ["A", "A&B", "B&C"])
    assert expression.tolist() == [[1.0, 1.0, 0.0], [3.0, 1.0, 1.0]]


def test_program_score_weights_by_squared_loading_and_skips_unscored_pairs():
    lr_pairs = np.array(["A^B", "C^D", "E^F"])
    matrix = csr_matrix(np.array([[1.0, 0.0]]))
    score = program_score(
        matrix, pd.Index(["A—B"]), np.array([2.0, 1.0, 0.0]), lr_pairs
    )
    assert score.tolist() == [4.0, 0.0]


def test_label_mean_scores_each_cell_at_its_types_mean():
    labels = np.array(["a", "a", "b", "b"])
    values = np.array([1.0, 0.0, 0.0, 0.0])
    assert label_mean(labels, values, np.array([True, False, True, False])) == 0.25


@pytest.fixture(name="cohort")
def fixture_cohort() -> Cohort:
    """Two samples of two types; NICHES marks the top of one weight."""
    rng = np.random.default_rng(0)
    n = 400
    weight = rng.normal(size=n)
    cells = {
        "Sample": np.tile(["s1", "s2"], n // 2),
        "Coarse": np.repeat(["a", "b"], n // 2),
    }
    cells["Fine"] = cells["Coarse"]
    for side, sign in (("Sender", 1), ("Receiver", -1)):
        cells[f"{side} C1"] = sign * weight
        cells[f"NICHES {side} C1"] = np.clip(sign * weight - 1, 0, None)
    return Cohort(pd.DataFrame(cells))


def test_populations_cut_each_weight_on_its_own():
    cells = pd.DataFrame(
        {"Sample": ["s"] * 3, "Coarse": ["a"] * 3, "Fine": ["a"] * 3}
        | {f"{side} C1": [1.0, 0.5, 0.1] for side in ("Sender", "Receiver")}
    )
    cohort = Cohort(cells)
    # Collinear weights: the same cells pass both cutoffs, in both populations
    for side in SIDES:
        assert cohort.population(1, side, tau=0.3).tolist() == [True, True, False]


def test_population_table_recovers_a_planted_population(cohort):
    table = population_table(cohort)
    assert table["NICHES+ cells"].gt(0).all()
    assert (table["Observed"] > table["Cell type predicted"]).all()
    assert table["Samples"].eq(2).all()


def test_cell_type_prediction_is_within_sample():
    # One type, two samples: NICHES+ only in s1, the population only in s1
    cells = pd.DataFrame(
        {
            "Sample": ["s1", "s1", "s2", "s2"],
            "Coarse": ["a"] * 4,
            "Fine": ["a"] * 4,
            "Sender C1": [1.0, 0.0, 0.0, 0.0],
            "Receiver C1": [1.0, 0.0, 0.0, 0.0],
            "NICHES Sender C1": [1.0, 0.0, 0.0, 0.0],
            "NICHES Receiver C1": [1.0, 0.0, 0.0, 0.0],
        }
    )
    row = population_table(Cohort(cells)).iloc[0]
    # Within s1 the type's rate is 1/2; across samples it would be 1/4
    assert row["Cell type predicted"] == 0.5


def test_tau_table_covers_every_tau_of_every_population(cohort):
    table = tau_table(cohort)
    assert table["Tau"].tolist() == list(TAUS) * len(SIDES)
    assert table["Side"].tolist() == [s for s in SIDES for _ in TAUS]
