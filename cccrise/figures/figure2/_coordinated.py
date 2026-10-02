"""Data shared by Figure 2's coordinated-change panels: the outcome-associated
CP component's LR pairs, scored on their own in the raw communication tensor."""

from functools import cache

import numpy as np

from .. import common
from ..figure3._shared import SPEC, association, factors_adata

DATASET = "alad"
# The CP component associated with ASSOCIATION (Figure 3), and its LR pairs
# with the largest weights
COMPONENT = 4
ASSOCIATION = "6-Month Outcome (4-Way)"
N_TOP = 10


def associations(matrix: np.ndarray):
    return common.column_associations(
        matrix, factors_adata(), association(ASSOCIATION), SPEC.condition_column
    )


@cache
def analysis() -> dict:
    """Per-LR associations, the whole component's, its top LR pairs, and
    their co-variation against `common.covariation_baselines`."""
    adata = factors_adata()
    scores = common.get_component_lr_scores(DATASET, COMPONENT)
    weights = np.asarray(adata.uns["D"])[:, COMPONENT - 1]
    top = np.argsort(-np.abs(weights))[:N_TOP]
    per_lr = associations(scores)
    component_eta = float(
        associations(np.asarray(adata.uns["A"])[:, [COMPONENT - 1]]).EtaSquared.iloc[0]
    )
    return {
        "lr_eta": per_lr.EtaSquared.to_numpy(),
        "component_eta": component_eta,
        "top": top,
        "observed": common.mean_pairwise_rank_correlation(scores, top),
        "baselines": common.covariation_baselines(
            scores, top, per_lr.PAdj.to_numpy() < common.SIG_ALPHA
        ),
    }
