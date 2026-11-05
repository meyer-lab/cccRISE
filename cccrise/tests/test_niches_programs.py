"""Unit tests for cccrise.comparators.niches_programs."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from cccrise.comparators.niches_programs import (
    CONTRIBUTION,
    EXPRESSION,
    RANDOM,
    RANDOM_EXPRESSION,
    SIDES,
    TAUS,
    WEIGHT,
    Cohort,
    centred_signal,
    centred_tokens,
    direction_draws,
    label_mean,
    ordering_table,
    pair_tokens,
    population_table,
    program_score,
    random_directions,
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


def test_centred_tokens_use_gene_means_but_a_complexs_own():
    expression = np.array([[1.0, 2.0], [3.0, 4.0]])
    centred = centred_tokens(expression, ["A", "A&B"], pd.Series({"A": 10.0}))
    assert centred.dtype == np.float32
    assert centred.tolist() == [[-9.0, -1.0], [-7.0, 1.0]]


def test_centred_signal_normalizes_over_present_tokens():
    centred = np.array([[1.0, 3.0], [2.0, 0.0]], dtype=np.float32)
    pair_side = pd.Series(["A", "B", "absent"])
    signal = centred_signal(centred, ["A", "B"], pair_side, np.array([1.0, 1.0, 5.0]))
    assert signal.tolist() == [2.0, 1.0]


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
    """Two types; NICHES marks the top of the first RISE axis, within each."""
    rng = np.random.default_rng(0)
    n = 400
    projections = rng.normal(size=(n, 4)).astype(np.float32)
    weight = projections[:, 0].astype(np.float64)
    cells = {"Coarse": np.repeat(["a", "b"], n // 2)}
    cells["Fine"] = cells["Coarse"]
    for side, sign in (("Sender", 1), ("Receiver", -1)):
        cells[f"{side} C1"] = sign * weight
        cells[f"Signal {side} C1"] = sign * weight
        cells[f"NICHES {side} C1"] = np.clip(sign * weight - 1, 0, None)
    return Cohort(pd.DataFrame(cells), projections)


def test_direction_draws_take_the_populations_cells_per_type(cohort):
    population = cohort.population(1, "sender")
    positive = cohort.column("NICHES", 1, "sender") > 0
    # A draw along the planted axis recovers the planted cells
    axis = np.zeros((4, 1), dtype=np.float32)
    axis[0] = 1
    assert direction_draws(cohort, positive, population, axis)[0] == pytest.approx(
        positive[population].mean()
    )


def test_populations_cut_each_weight_on_its_own():
    cells = pd.DataFrame(
        {"Coarse": ["a"] * 3, "Fine": ["a"] * 3}
        | {f"{side} C1": [1.0, 0.5, 0.1] for side in ("Sender", "Receiver")}
    )
    cohort = Cohort(cells, np.zeros((3, 2), np.float32))
    # Collinear weights: the same cells pass both cutoffs, in both populations
    for side in SIDES:
        assert cohort.population(1, side, tau=0.3).tolist() == [True, True, False]


def test_population_table_recovers_a_planted_population(cohort):
    directions = random_directions(4, 50, 0)
    table = population_table(cohort, directions)
    assert table["Scorable"].all()
    assert (table["Observed"] > table["Labels fine"]).all()
    assert table["Score fold"].gt(1).all()
    assert table["Q direction"].notna().all()


def test_ordering_table_has_every_ordering_and_is_seeded(cohort):
    directions = random_directions(4, 20, 0)
    table = ordering_table(cohort, directions)
    orderings = {RANDOM, WEIGHT, RANDOM_EXPRESSION, CONTRIBUTION, EXPRESSION}
    assert set(table["Ordering"]) == orderings
    shares = table.set_index(["Side", "Ordering"])["Share"]
    # Expression takes every NICHES+ cell first, here fewer than the population
    positive = cohort.column("NICHES", 1, "sender") > 0
    n = cohort.population(1, "sender").sum()
    assert shares["sender", EXPRESSION] == pytest.approx(positive.sum() / n)
    pd.testing.assert_frame_equal(
        table, ordering_table(cohort, random_directions(4, 20, 0))
    )


def test_tau_table_covers_every_tau_of_scorable_populations(cohort):
    populations = population_table(cohort, random_directions(4, 20, 0))
    populations.loc[populations["Side"] == "receiver", "Scorable"] = False
    table = tau_table(cohort, populations)
    assert table["Tau"].tolist() == list(TAUS)
    assert set(table["Side"]) == {"sender"}
