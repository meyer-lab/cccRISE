"""Phenotype AUC as a function of CP rank, for one method.

Reviewer #1's comment #4 is that the ALAD ranks were chosen because they
"yielded the optimal predictive performance for ALAD status", so the reported
AUC is optimistically biased. The ranks are now selected by BiCV and CORCONDIA,
which never see the phenotype -- but asserting that the bias is gone is weaker
than showing what it was. This module produces the curve that shows it: AUC
against CP rank, with the rank the registry actually selected marked on it.

It also explains the first benchmark result (`.deferred/BENCHMARK_PLAN.md`
§0.2), where CCC-RISE scored 0.485 at CP 3. If the curve climbs toward the much
higher ranks the manuscript originally used, that is the bias being measured
rather than argued about.

What is held fixed
------------------
RISE is fit **once** per method, and every CP rank decomposes that same
interaction tensor -- the same discipline `run_cpd_rank` uses, for the same
reason: a curve whose points differ in their RISE fit is not a curve in CP rank.

Each method's CP stage is called with *its own pipeline's* settings rather than
a single shared set, so the point at the selected rank reproduces that method's
published number instead of approximating it. That is what makes the curve
connect to the committed benchmark table: CCC-RISE at rank 3 here is the model
in `bal_alad_factors.nc`, not a near-miss of it.

CCC-RISE standardizes its CP factors (`standardize_cc_pf2`) and the pseudobulk
workflow does not; both are mirrored as-is. The AUC is unaffected either way --
`cv_auc` scales every column inside its folds, and neither component order nor
component sign changes a cross-validated AUC -- but mirroring keeps each point
on the curve the same object its pipeline produces.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import anndata
import numpy as np
import pandas as pd
from tensorly.decomposition import parafac

from ..ccc_rise import standardize_cc_pf2, validate_cp_rank
from ..import_data import add_cond_idxs
from ..tensor import calculate_interaction_tensor, rise_store_r2x
from ..utils import pseudobulk_X
from .registry import MethodScores, _condition_order, _corrected_condition_factor


@dataclass
class RankSweepTensor:
    """One method's interaction tensor, plus everything needed to score a CP fit
    of it at any rank."""

    key: str
    label: str
    tensor: np.ndarray
    adata: anndata.AnnData
    condition_column: str
    #: CP settings this method's own pipeline uses.
    cp_n_iter_max: int
    cp_tol: float
    standardize: bool
    random_state: int
    extra: dict

    @property
    def max_rank(self) -> int:
        """CP rank is capped by the tensor's smallest mode.

        This is the structural asymmetry between the two methods, and it is a
        result in its own right: the pseudobulk tensor's sender and receiver
        modes are cell types, so it cannot carry more components than there are
        cell types (8 on ALAD's `broad_cell_type`). CCC-RISE's are eigen-states,
        so it runs to the RISE rank (19). The manuscript's original ALAD
        classifier used components numbered up to 17 -- a rank the cell-type
        tensor cannot reach at all.
        """
        return int(min(self.tensor.shape))

    def usable_ranks(self, ranks: list[int]) -> list[int]:
        return [r for r in ranks if r <= self.max_rank]

    def scores_at(self, rank: int) -> MethodScores:
        """Fit CP at `rank` and return the read-depth corrected condition factor."""
        validate_cp_rank(rank, self.tensor.shape)
        weights, factors = parafac(
            self.tensor,
            rank,
            n_iter_max=self.cp_n_iter_max,
            tol=self.cp_tol,
            init="svd",
            svd="randomized_svd",
            normalize_factors=True,
            random_state=self.random_state,
        )
        if self.standardize:
            weights, factors = standardize_cc_pf2(weights, factors)

        X = self.adata
        X.uns["A"], X.uns["B"], X.uns["C"], X.uns["D"] = factors
        A = _corrected_condition_factor(X)
        return MethodScores(
            scores=A,
            condition_names=_condition_order(X, self.condition_column),
            rank=rank,
            extra={"cp_weight_sum": float(np.sum(weights))},
        )

    def iter_scores(self, ranks: list[int]) -> Iterator[tuple[int, MethodScores]]:
        for rank in self.usable_ranks(ranks):
            yield rank, self.scores_at(rank)


def ccc_rise_sweep_tensor(
    adata: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    condition_column: str,
    rise_rank: int,
    rise_n_iter_max: int = 300,
    rise_tol: float = 1e-8,
    complex_sep: str | None = "&",
    random_state: int = 42,
) -> RankSweepTensor:
    """Fit RISE once and build the eigen-state interaction tensor.

    Defaults are the published pipeline's per-stage convergence settings, so the
    tensor is the one `bal_alad_factors.nc` was decomposed from. Loosening them
    would make every point on the curve a different model from the committed
    one.
    """
    adata = add_cond_idxs(adata, condition_column)
    adata, r2x_pf2 = rise_store_r2x(
        adata,
        rank=rise_rank,
        n_iter_max=rise_n_iter_max,
        tolerance=rise_tol,
        random_state=random_state,
    )
    tensor, filtered_lr_pairs = calculate_interaction_tensor(
        adata, lr_pairs=lr_pairs, complex_sep=complex_sep
    )
    return RankSweepTensor(
        key="ccc_rise",
        label="CCC-RISE",
        tensor=tensor,
        adata=adata,
        condition_column=condition_column,
        cp_n_iter_max=4000,
        cp_tol=1e-12,
        standardize=True,
        random_state=random_state,
        extra={
            "rise_rank": rise_rank,
            "rise_n_iter_max": rise_n_iter_max,
            "rise_tol": rise_tol,
            "r2x_pf2": float(r2x_pf2),
            "tensor_shape": list(tensor.shape),
            "n_lr_pairs": len(filtered_lr_pairs),
        },
    )


def pseudobulk_sweep_tensor(
    adata: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    condition_column: str,
    groupby: str,
    complex_sep: str | None = "&",
    aggregation: str = "mean",
    random_state: int = 42,
) -> RankSweepTensor:
    """Aggregate to cell types once and build the pseudobulk tensor.

    CP settings mirror `tensor.run_pseudobulk_workflow`'s defaults
    (`n_iter_max=1000, tol=1e-11`), which is what the committed benchmark row
    for this method used.
    """
    from ..ccc_rise import calc_communication_score_pseudobulk

    adata = add_cond_idxs(adata, condition_column)
    matrices = pseudobulk_X(
        adata, condition_name=condition_column, groupby=groupby, type=aggregation
    )
    tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(
        matrices, lr_pairs=lr_pairs, complex_sep=complex_sep
    )
    return RankSweepTensor(
        key="pseudobulk_cp",
        label="Pseudobulk CPD",
        tensor=tensor,
        adata=adata,
        condition_column=condition_column,
        cp_n_iter_max=1000,
        cp_tol=1e-11,
        standardize=False,
        random_state=random_state,
        extra={
            "groupby": groupby,
            "aggregation": aggregation,
            "tensor_shape": list(tensor.shape),
            "n_lr_pairs": len(filtered_lr_pairs),
            "n_cell_types": int(tensor.shape[1]),
        },
    )


#: Builders keyed the same way as `registry.METHODS`, so a CLI can name either.
SWEEP_BUILDERS = {
    "ccc_rise": ccc_rise_sweep_tensor,
    "pseudobulk_cp": pseudobulk_sweep_tensor,
}
