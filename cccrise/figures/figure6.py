"""
Figure 6: CP Rank Selection Analysis

This figure demonstrates how R²X changes with increasing CP rank
for the COVID dataset using cached RISE factors.
"""

import pandas as pd
from matplotlib import pyplot as plt
from tlviz.model_evaluation import core_consistency

from ..ccc_rise import calculate_interaction_tensor, parafac_multistart
from ..import_data import import_ligand_receptor_pairs
from ..rank_selection import calculate_r2x
from .common import get_covid_data, subplotLabel


def compute_cp_rank_sweep(
    cp_ranks: list[int] | None = None,
    n_iter_max: int = 200,
    tol: float = 1e-3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute R2X and core consistency across CP ranks for cached RISE factors of the COVID dataset."""
    if cp_ranks is None:
        cp_ranks = list(range(1, 13, 1))

    adata = get_covid_data(project_cells=True)
    lr_pairs = import_ligand_receptor_pairs()

    # Calculate interaction tensor ONCE for the cached RISE dataset
    interaction_tensor, _ = calculate_interaction_tensor(adata, lr_pairs)

    r2x_results = []
    for cp_rank in cp_ranks:
        print(f"CP rank: {cp_rank}")
        cp_weights, cp_factors = parafac_multistart(
            interaction_tensor,
            cp_rank,
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=random_state,
        )
        r2x = calculate_r2x(cp_weights, cp_factors, interaction_tensor)
        cc = core_consistency(
            (cp_weights, cp_factors), interaction_tensor, normalised=True
        )
        print(f"R2X: {r2x}, Core Consistency: {cc}")
        r2x_results.append({"cp_rank": cp_rank, "r2x": r2x, "core_consistency": cc})

    return pd.DataFrame(r2x_results)


def makeFigure():
    """Generate Figure 6 showing CP rank selection analysis with cached RISE factors."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    subplotLabel([ax])

    df = compute_cp_rank_sweep()

    ax.plot(
        df["cp_rank"],
        df["r2x"],
        marker="o",
        color="blue",
        label="R²X",
    )
    ax.plot(
        df["cp_rank"],
        df["core_consistency"] / 100.0,
        marker="o",
        color="orange",
        label="Core Consistency",
    )
    ax.set_xlabel("CP Rank")
    ax.set_ylabel("Score")
    ax.set_title("CP Rank Selection")
    ax.legend()

    return fig
