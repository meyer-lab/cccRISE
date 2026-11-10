"""cccRISE's sender and receiver populations against upstream NICHES.

NICHES (msraredon/NICHES, R; `niches_system.R`) scores every cell against its
sample: CellToSystem as a sender, SystemToCell as a receiver. A component's
*program score* sums those scores over its carrying pairs
(`covariation.carrying_pairs`), weighted by the squared LR loading. A cell is
NICHES+ when that score is non-zero, and a set of cells' *NICHES concordance*
is its share of NICHES+ cells.

A component's populations are label-free: its sender population is the cells
whose sender weight is at least `GROUP_TAU` x its largest value, and its
receiver population likewise on the receiver weight (`Cohort.population`). A
cell past both cutoffs is in both, so a component whose sender and receiver
weights are collinear is not split arbitrarily. Two tables follow them, each
against what cell type predicts *within sample*: each population cell at the
NICHES+ rate of its (sample, fine type) stratum (`Cohort.strata`), since NICHES
scores every cell against its own sample:

- `population_table`: each population's concordance and that prediction.
- `tau_table`: the same across `TAUS`.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

from ..covariation import carrying_pairs
from ..figures.figure2._common import GROUP_TAU

TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
SIDES = ("sender", "receiver")
#: Each side's NICHES mode
MODES = {"sender": "c2s", "receiver": "s2c"}
COMPLEX_SEP = "&"


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
    """Every cell NICHES scored. `cells` has ``Sample``, ``Fine`` and, per side
    and component (``Sender C1``, ...), the weight and its ``NICHES`` column."""

    cells: pd.DataFrame
    strata: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.strata = (
            self.cells["Sample"].astype(str) + "\x1f" + self.cells["Fine"].astype(str)
        ).to_numpy()

    @property
    def n_components(self) -> int:
        return sum(c.startswith("Sender C") for c in self.cells.columns)

    def population(self, k: int, side: str, tau: float = GROUP_TAU) -> np.ndarray:
        """Mask of component `k`'s `side` population at `tau`."""
        return in_population(self.column("", k, side), tau)

    def column(self, kind: str, k: int, side: str) -> np.ndarray:
        """A per-cell column: `kind` is "" (the weight) or "NICHES"."""
        return self.cells[f"{kind} {side.title()} C{k}".strip()].to_numpy()


def in_population(weight: np.ndarray, tau: float = GROUP_TAU) -> np.ndarray:
    """Cells whose `weight` is at least `tau` x its largest value."""
    return weight >= tau * weight.max()


def label_mean(labels: np.ndarray, values: np.ndarray, cells: np.ndarray) -> float:
    """What labels predict for the mean of `values` over `cells`: each cell at
    its label's mean."""
    means = pd.Series(values).groupby(labels).mean()
    return float(means.reindex(labels[cells]).mean())


def _row(cohort: Cohort, k: int, side: str, tau: float) -> dict[str, Any]:
    population = cohort.population(k, side, tau)
    positive = cohort.column("NICHES", k, side) > 0
    return {
        "Component": k,
        "Side": side,
        "Cells": int(population.sum()),
        "Samples": int(cohort.cells.loc[population, "Sample"].nunique()),
        "NICHES+ cells": int(positive[population].sum()),
        "Observed": float(positive[population].mean()),
        "Cell type predicted": label_mean(cohort.strata, positive, population),
    }


def population_table(cohort: Cohort) -> pd.DataFrame:
    """Each population's concordance (``Observed``) and what cell type predicts
    within sample (module docstring), at `GROUP_TAU`."""
    return pd.DataFrame(
        [
            _row(cohort, k, side, GROUP_TAU)
            for k in range(1, cohort.n_components + 1)
            for side in SIDES
        ]
    )


def tau_table(cohort: Cohort) -> pd.DataFrame:
    """`population_table` at every tau of `TAUS`."""
    return pd.DataFrame(
        [
            _row(cohort, k, side, tau) | {"Tau": tau}
            for k in range(1, cohort.n_components + 1)
            for side in SIDES
            for tau in TAUS
        ]
    )
