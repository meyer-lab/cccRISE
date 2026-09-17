"""Registry of benchmarked methods, one entry per method being compared.

`run_benchmark_auc` iterates this and scores whatever it gets back through
`auc.cv_auc`; adding a method to the benchmark is adding an entry here, not
editing a runner. That is the point -- a method that needs bespoke handling in
the runner to look good is a method whose advantage is in the handling.

The contract on the tensor axis is deliberately narrow. A method is a callable
from `MethodInputs` to a `MethodScores`, whose `scores` is a
`(n_conditions, n_components)` matrix in `np.unique(obs[condition_column])`
order -- the same order `plotGeneral.get_sample_to_group` returns labels in, so
rows and labels line up without either side reindexing the other.

Read-depth correction (`utils.correct_conditions`) is applied to **every**
method's condition scores, not just CCC-RISE's. It is a property of the cohort
-- ALAD's per-patient totals span a 45x range -- not of any one decomposition,
and correcting one method's factor but not another's would be the kind of
asymmetry a referee finds first.

Currently registered are the two methods that need no new algorithm: CCC-RISE
itself, read from the published factor model, and the pseudobulk CP baseline the
manuscript already reports. Tensor-cell2cell and STACCato land in PR 2, and the
single-cell axis in PR 3.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd

from ..factors import load_factors_to_anndata
from ..figures.commonFuncs.plotGeneral import get_sample_to_group
from ..runners.datasets import DatasetSpec
from ..utils import correct_conditions

#: Axis labels, matching `.deferred/BENCHMARK_PLAN.md`.
TENSOR_AXIS = "tensor"
SINGLE_CELL_AXIS = "single_cell"


@dataclass(frozen=True)
class MethodInputs:
    """Everything a method may need, resolved once and shared by all of them.

    Loading ALAD is the expensive part of the runner, so it happens once and
    every method sees the identical `adata` -- same cells, same genes, same
    normalization. A method that re-loaded or re-filtered would no longer be
    decomposing the same data as its neighbours in the table.
    """

    spec: DatasetSpec
    adata: anndata.AnnData
    lr_pairs: pd.DataFrame
    factors_path: Path
    rise_rank: int
    cp_rank: int
    random_state: int = 42


@dataclass(frozen=True)
class MethodScores:
    """One method's per-condition scores, ready for `auc.cv_auc`."""

    scores: np.ndarray
    condition_names: np.ndarray
    rank: int
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scores.ndim != 2:
            raise ValueError(
                f"scores must be (n_conditions, n_components), got {self.scores.shape}"
            )
        if len(self.condition_names) != self.scores.shape[0]:
            raise ValueError(
                f"{self.scores.shape[0]} score rows but "
                f"{len(self.condition_names)} condition names"
            )


@dataclass(frozen=True)
class BenchmarkMethod:
    key: str
    label: str
    axis: str
    fit: Callable[[MethodInputs], MethodScores]
    notes: str = ""


def _corrected_condition_factor(X: anndata.AnnData) -> np.ndarray:
    """`uns["A"]` scaled by the read-depth fit, as every downstream analysis in
    this package uses it (`anova.prepare_anova_inputs` does the same)."""
    return np.asarray(correct_conditions(X), dtype=float)


def _condition_order(adata: anndata.AnnData, condition_column: str) -> np.ndarray:
    """The row order `uns["A"]` is in, and that labels must be fetched in."""
    return np.unique(adata.obs[condition_column])


def ccc_rise_scores(inputs: MethodInputs) -> MethodScores:
    """CCC-RISE's condition factor, from the published factor model.

    Reads the committed `.nc` rather than refitting: that file *is* the model the
    manuscript reports, and a benchmark that quietly refits it at different
    convergence settings would be scoring a different model than the one under
    review (`.deferred/BENCHMARK_PLAN.md` pitfall 1).
    """
    from ..factors import load_factors_netcdf

    factors_ds = load_factors_netcdf(inputs.factors_path)

    # The runner compares each method's reported condition order against the
    # labels, but for every other method that order is *derived* from the same
    # AnnData the labels come from, so the check cannot fail. Here it can: the
    # `.nc` carries its own condition coordinate, written by whichever run
    # produced it. A factor file fit on a different roster -- a stale copy, a
    # subset, another cohort -- would otherwise be silently paired with this
    # cohort's labels row by row.
    stored = np.asarray(factors_ds.coords["condition"].values).astype(str)
    expected = _condition_order(inputs.adata, inputs.spec.condition_column).astype(str)
    if not np.array_equal(stored, expected):
        raise ValueError(
            f"{inputs.factors_path} was fit on a different set of conditions than "
            f"the loaded dataset: {len(stored)} stored vs {len(expected)} loaded, "
            f"first mismatch at "
            f"{next((i for i, (a, b) in enumerate(zip(stored, expected)) if a != b), 'length')}. "
            "Regenerate it with `uv run cccrise-run` before benchmarking."
        )

    X = load_factors_to_anndata(
        factors_ds,
        adata=inputs.adata.copy(),
        condition_column=inputs.spec.condition_column,
    )
    A = _corrected_condition_factor(X)
    return MethodScores(
        scores=A,
        condition_names=_condition_order(X, inputs.spec.condition_column),
        rank=int(A.shape[1]),
        extra={
            "source": str(inputs.factors_path),
            "rise_rank": inputs.rise_rank,
            "r2x_cp": float(X.uns.get("r2x_cp", np.nan)),
        },
    )


def pseudobulk_cp_scores(inputs: MethodInputs) -> MethodScores:
    """The pseudobulk CP baseline the manuscript already compares against.

    This is the "pseudobulk CPD equivalent" Reviewer #1 objected to as the only
    benchmark. It stays in the table: the objection was that it was the *sole*
    comparison, not that it is uninformative, and it is the natural bridge
    between CCC-RISE and a real cell-type tensor method. It aggregates to
    `celltype_column` before scoring, so it is signed, unmasked CP on an
    8-cell-type tensor -- which is what makes it a baseline rather than a
    re-implementation of Tensor-cell2cell (see PR 2).
    """
    from ..tensor import run_pseudobulk_workflow

    groupby = inputs.spec.celltype_column or "celltype"
    X, r2x = run_pseudobulk_workflow(
        inputs.adata.copy(),
        cp_rank=inputs.cp_rank,
        lr_pairs=inputs.lr_pairs,
        condition_column=inputs.spec.condition_column,
        groupby=groupby,
        random_state=inputs.random_state,
    )
    A = _corrected_condition_factor(X)
    return MethodScores(
        scores=A,
        condition_names=_condition_order(X, inputs.spec.condition_column),
        rank=int(A.shape[1]),
        extra={"groupby": groupby, "r2x_cp": float(r2x)},
    )


METHODS: dict[str, BenchmarkMethod] = {
    "ccc_rise": BenchmarkMethod(
        key="ccc_rise",
        label="CCC-RISE",
        axis=TENSOR_AXIS,
        fit=ccc_rise_scores,
        notes="Published factor model; eigen-state tensor, signed CP.",
    ),
    "pseudobulk_cp": BenchmarkMethod(
        key="pseudobulk_cp",
        label="Pseudobulk CPD",
        axis=TENSOR_AXIS,
        fit=pseudobulk_cp_scores,
        notes="Cell-type pseudobulk tensor, signed CP. The manuscript's own baseline.",
    ),
}

METHOD_KEYS = tuple(METHODS)


def resolve_methods(keys: list[str] | None = None) -> list[BenchmarkMethod]:
    """Look up methods by key, refusing to silently skip an unknown one."""
    if not keys:
        return list(METHODS.values())
    unknown = [k for k in keys if k not in METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}. Registered: {sorted(METHODS)}")
    return [METHODS[k] for k in keys]


def method_labels(
    inputs: MethodInputs,
    group_column: str,
) -> pd.Series:
    """The phenotype label per condition, in the score matrices' row order.

    Every method is scored against this same Series, fetched once, so a
    difference between two rows of the benchmark table can only come from the
    scores.
    """
    return get_sample_to_group(inputs.adata, inputs.spec.condition_column, group_column)
