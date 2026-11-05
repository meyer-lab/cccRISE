"""cccRISE's sender and receiver populations against upstream NICHES.

NICHES (msraredon/NICHES, R; `niches_system.R`) scores every cell against its
sample: CellToSystem as a sender, SystemToCell as a receiver. A component's
*program score* sums those scores over its carrying pairs
(`covariation.carrying_pairs`), weighted by the squared LR loading. A cell is
NICHES+ when that score is non-zero, and a set of cells' *NICHES concordance*
is its share of NICHES+ cells.

A component's populations are Figure 2's (`figure2._common.split_cells`,
label-free). Three tables follow them:

- `population_table`: each population's concordance against what its own label
  composition predicts (each cell at its type's rate), and against random RISE
  directions taking the same number of cells from each fine type. Its score
  fold is the same comparison on the mean program score.
- `ordering_table`: the population's number of cells chosen by a state
  direction, cccRISE's or a random one, alone and times expression.
  Multiplying by expression reaches NICHES from either direction, since
  expression is what NICHES reads.
- `tau_table`: a population's concordance over its labels' across `TAUS`.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import false_discovery_control

from ..covariation import carrying_pairs
from ..figures.figure2._common import GROUP_TAU, split_cells

#: A population is scorable when this share of its cells is NICHES+
SCORABLE = 0.10
#: Draws of random directions: against a population, and in `ordering_table`
N_POPULATION_DRAWS = 1000
N_ORDERING_DRAWS = 200
TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
SIDES = ("sender", "receiver")
#: Each side's NICHES mode
MODES = {"sender": "c2s", "receiver": "s2c"}
COMPLEX_SEP = "&"
#: `ordering_table`'s orderings
WEIGHT, CONTRIBUTION, EXPRESSION = "Weight", "Contribution", "Expression"
RANDOM, RANDOM_EXPRESSION = "Random direction", "Random direction x expression"


def pair_tokens(lr_pairs: np.ndarray) -> pd.DataFrame:
    """The `ligand` and `receptor` token of each ``ligand^receptor`` name."""
    return (
        pd.Series(lr_pairs)
        .str.split("^", expand=True)
        .set_axis(["ligand", "receptor"], axis=1)
    )


def token_expression(X: Any, genes: pd.Index, tokens: list[str]) -> np.ndarray:
    """(cells x tokens) expression of `tokens`, a complex as the minimum over
    its subunits (`ccc_rise.add_complexes_to_expression`). Every subunit must
    be in `genes`."""
    needed = sorted({g for t in tokens for g in t.split(COMPLEX_SEP)})
    position = {g: i for i, g in enumerate(needed)}
    columns = csc_matrix(csr_matrix(X)[:, genes.get_indexer(needed)])
    out = np.empty((columns.shape[0], len(tokens)), columns.dtype)
    for j, token in enumerate(tokens):
        subunits = [position[g] for g in token.split(COMPLEX_SEP)]
        out[:, j] = columns[:, subunits].toarray().min(axis=1)
    return out


def centred_tokens(
    expression: np.ndarray, tokens: list[str], gene_means: pd.Series
) -> np.ndarray:
    """`expression` as float32, centred as RISE centres it: a gene on the
    fit's gene mean, a complex (no fit mean) on its own mean over cells."""
    centred = np.empty(expression.shape, np.float32)
    for j, token in enumerate(tokens):
        mean = expression[:, j].mean() if COMPLEX_SEP in token else gene_means[token]
        centred[:, j] = expression[:, j] - mean
    return centred


def centred_signal(
    centred: np.ndarray, tokens: list[str], pair_side: pd.Series, loadings: np.ndarray
) -> np.ndarray:
    """Per-cell signal of one component and side (#241): each token's `centred`
    expression weighted by its pairs' squared `loadings`, normalized over the
    `tokens` present. `pair_side` holds each LR pair's token of that side."""
    weights = pd.Series(loadings**2, index=pair_side.to_numpy()).groupby(level=0).sum()
    weights = weights.reindex(tokens).fillna(0.0)
    return centred.astype(np.float64) @ (weights / weights.sum()).to_numpy()


def program_score(
    matrix: Any, rows: pd.Index, loadings: np.ndarray, lr_pairs: np.ndarray
) -> np.ndarray:
    """A component's program score from one NICHES `matrix` (LR pair x cell,
    its pairs named by `rows`), over the carrying pairs NICHES scored."""
    carrying = carrying_pairs(loadings)
    names = [lr_pairs[i].replace("^", "—") for i in carrying]
    found = rows.get_indexer(names)
    ok = found >= 0
    return np.asarray(matrix[found[ok]].T @ (loadings[carrying][ok] ** 2)).ravel()


@dataclass
class Cohort:
    """Every cell NICHES scored. `cells` has ``Coarse``, ``Fine`` and, per side
    and component (``Sender C1``, ...), the weight and its ``Signal`` and
    ``NICHES`` columns; `projections` their RISE projections."""

    cells: pd.DataFrame
    projections: np.ndarray
    fine: np.ndarray = field(init=False)
    coarse: np.ndarray = field(init=False)
    types: np.ndarray = field(init=False)
    type_index: np.ndarray = field(init=False)
    members: list[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.fine = self.cells["Fine"].to_numpy()
        self.coarse = self.cells["Coarse"].to_numpy()
        self.types, self.type_index = np.unique(self.fine, return_inverse=True)
        self.members = [
            np.flatnonzero(self.type_index == i) for i in range(len(self.types))
        ]

    @property
    def n_components(self) -> int:
        return sum(c.startswith("Sender C") for c in self.cells.columns)

    def population(self, k: int, side: str, tau: float = GROUP_TAU) -> np.ndarray:
        """Mask of component `k`'s `side` population at `tau`."""
        sender = self.cells[f"Sender C{k}"].to_numpy()
        receiver = self.cells[f"Receiver C{k}"].to_numpy()
        return split_cells(sender, receiver, tau)[side]

    def column(self, kind: str, k: int, side: str) -> np.ndarray:
        """A per-cell column: `kind` is "" (the weight), "Signal" or "NICHES"."""
        return self.cells[f"{kind} {side.title()} C{k}".strip()].to_numpy()


def random_directions(n_eigenstates: int, n: int, random_state: int) -> np.ndarray:
    """`n` directions uniform on the sphere, as columns."""
    rng = np.random.default_rng(random_state)
    directions = rng.normal(size=(n_eigenstates, n)).astype(np.float32)
    return directions / np.linalg.norm(directions, axis=0)


def label_mean(labels: np.ndarray, values: np.ndarray, cells: np.ndarray) -> float:
    """What labels predict for the mean of `values` over `cells`: each cell at
    its type's mean."""
    means = pd.Series(values).groupby(labels).mean()
    return float(means.reindex(labels[cells]).mean())


def direction_draws(
    cohort: Cohort, positive: np.ndarray, population: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Concordance of each direction's top cells, taking as many cells from
    each fine type as `population` does."""
    counts = np.bincount(cohort.type_index[population], minlength=len(cohort.types))
    hits = np.zeros(directions.shape[1])
    for i in np.flatnonzero(counts):
        cells, n = cohort.members[i], counts[i]
        if n >= len(cells):
            hits += positive[cells].sum()
            continue
        order = cohort.projections[cells] @ directions
        top = np.argpartition(order, len(cells) - n, axis=0)[len(cells) - n :]
        hits += positive[cells][top].sum(axis=0)
    return hits / population.sum()


def population_row(
    cohort: Cohort, k: int, side: str, directions: np.ndarray
) -> dict[str, Any]:
    """One population against its labels and, when scorable, random directions."""
    population = cohort.population(k, side)
    score = cohort.column("NICHES", k, side)
    positive = score > 0
    observed = float(positive[population].mean())
    score_observed = float(score[population].mean())
    score_labels = label_mean(cohort.fine, score, population)
    row = {
        "Component": k,
        "Side": side,
        "Cells": int(population.sum()),
        "Observed": observed,
        "Labels fine": label_mean(cohort.fine, positive, population),
        "Labels coarse": label_mean(cohort.coarse, positive, population),
        "Score observed": score_observed,
        "Score labels fine": score_labels,
        "Score fold": score_observed / score_labels,
        "Scorable": observed >= SCORABLE,
    }
    if row["Scorable"]:
        null = direction_draws(cohort, positive, population, directions)
        row |= {
            "Direction 5th": float(np.quantile(null, 0.05)),
            "Direction 95th": float(np.quantile(null, 0.95)),
            "P direction": float((1 + (null >= observed).sum()) / (1 + len(null))),
        }
    return row


def population_table(cohort: Cohort, directions: np.ndarray) -> pd.DataFrame:
    """Every population (`population_row`), BH q over the scorable ones."""
    table = pd.DataFrame(
        [
            population_row(cohort, k, side, directions)
            for k in range(1, cohort.n_components + 1)
            for side in SIDES
        ]
    )
    scorable = table["Scorable"]
    table.loc[scorable, "Q direction"] = false_discovery_control(
        table.loc[scorable, "P direction"]
    )
    return table


def top_cells(order: np.ndarray, n: int) -> np.ndarray:
    return np.argpartition(order, len(order) - n)[len(order) - n :]


def ordering_shares(
    cohort: Cohort, k: int, side: str, directions: np.ndarray
) -> dict[str, float]:
    """Concordance of the population's number of cells under each ordering."""
    score = cohort.column("NICHES", k, side)
    weight = cohort.column("", k, side)
    signal = cohort.column("Signal", k, side)
    positive, n = score > 0, int(cohort.population(k, side).sum())

    def share(order: np.ndarray) -> float:
        return float(positive[top_cells(order, n)].mean())

    random = cohort.projections @ directions
    return {
        RANDOM: float(np.median([share(d) for d in random.T])),
        WEIGHT: share(weight),
        RANDOM_EXPRESSION: float(
            np.median([share(np.clip(d, 0, None) * signal) for d in random.T])
        ),
        CONTRIBUTION: share(np.clip(weight, 0, None) * signal),
        EXPRESSION: share(signal),
    }


def ordering_table(cohort: Cohort, directions: np.ndarray) -> pd.DataFrame:
    """Every component and side's `ordering_shares`, and how sparse NICHES is
    on it: its share of all cells."""
    rows = []
    for k in range(1, cohort.n_components + 1):
        for side in SIDES:
            shares = ordering_shares(cohort, k, side, directions)
            cohort_share = float((cohort.column("NICHES", k, side) > 0).mean())
            rows += [
                {
                    "Component": k,
                    "Side": side,
                    "Ordering": ordering,
                    "Share": share,
                    "Cohort share": cohort_share,
                }
                for ordering, share in shares.items()
            ]
    return pd.DataFrame(rows)


def tau_table(cohort: Cohort, populations: pd.DataFrame) -> pd.DataFrame:
    """Each scorable population's concordance and its labels' across `TAUS`."""
    rows = []
    scorable = populations[populations["Scorable"]]
    for k, side in zip(scorable["Component"], scorable["Side"], strict=True):
        positive = cohort.column("NICHES", k, side) > 0
        for tau in TAUS:
            population = cohort.population(k, side, tau)
            rows.append(
                {
                    "Component": k,
                    "Side": side,
                    "Tau": tau,
                    "Cells": int(population.sum()),
                    "Observed": float(positive[population].mean()),
                    "Labels fine": label_mean(cohort.fine, positive, population),
                }
            )
    return pd.DataFrame(rows)
