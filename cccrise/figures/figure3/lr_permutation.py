"""Patient-permutation test of each CP component's top LR pairs between
CLAD/Died and No CLAD samples, supporting the outcome association of the
component this figure follows.

Each LR pair is scored per condition in each component's own
sender/receiver context (`coordinated_changes.component_lr_scores`), divided
by the condition's read depth (as the plotted condition loadings are) and
ranked across conditions, so no single sample's magnitude drives the test.
The CLAD/Died minus No CLAD mean rank difference is compared against
`N_PERMS` permutations of the outcome labels by patient
(`patient_permutation_differential`), each difference standardized by its
null standard deviation. A component's set statistic is the mean
|standardized difference| of its `N_TOP` most heavily weighted pairs, and
its set p the fraction of permutations reaching it.

The panel is each component's permutation null of the set statistic beside
the observed statistic; the other components are the specificity control.
`figureS_lr_permutation` varies the set size and shows the individual pairs.
"""

from functools import cache
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ...anova import get_sample_to_group, prepare_group_labels
from ...factors import load_factors_netcdf
from ...validation.significance import patient_permutation_differential
from .. import common, style
from ._common import COMPONENT, DATASET, SPEC, association, factors_adata
from .coordinated_changes import N_TOP, component_lr_scores

CONTRAST = "6-Month CLAD/Died vs. No CLAD"
CASE_LABEL = "CLAD/Died"
N_PERMS = 10_000
MAX_TOP = 30  # largest top-pair set the set test is run on
AXES = ("square",)


def plot(axs):
    plot_set_null_violins(permutation_analysis(), N_TOP, axs[0])


def plot_set_null_violins(result: dict[int, dict], n_top: int, ax) -> None:
    """Per component, the permutation null of its top-`n_top` set statistic
    (violin, its 95th percentile dashed) and the observed statistic
    (diamond), labeled with the set p."""
    components = sorted(result)
    nulls = [set_statistic(result[k]["null_z"], n_top) for k in components]
    bodies = ax.violinplot(nulls, positions=components, showextrema=False, widths=0.8)
    for body in bodies["bodies"]:
        body.set(facecolor=style.COLORS["muted"], edgecolor="none", alpha=0.8)
    for k, null in zip(components, nulls, strict=True):
        observed = set_statistic(result[k]["z"][result[k]["order"]], n_top)
        critical = np.quantile(null, 1 - common.SIG_ALPHA)
        ax.hlines(
            critical,
            k - 0.3,
            k + 0.3,
            color=style.COLORS["reference"],
            lw=style.LINE_WIDTH_THIN,
            linestyle=style.LINE_STYLE_REFERENCE,
        )
        ax.scatter(
            k,
            observed,
            marker="D",
            s=style.POINT_SIZE_FOREGROUND,
            color=style.COLORS["emphasis" if k == COMPONENT else "ink"],
            zorder=3,
        )
        ax.text(
            k,
            max(observed, critical) + 0.05,
            f"{set_pvalue(result[k], n_top):.2g}",
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
        )
    ax.set_xticks(components)
    ax.set(
        xlabel="Component",
        ylabel=f"Top-{n_top} LR pair\nmean |diff.| (null SDs)",
    )


@cache
def permutation_analysis() -> dict[int, dict]:
    """Per component, from its own context scores: every LR pair's
    standardized difference (`z`) and two-sided permutation p (`p`), the
    pairs ordered by |loading| (`order`), each group's median score
    (`medians`), and the standardized null differences of its `MAX_TOP`
    heaviest pairs (`null_z`, permutations x pairs) from which `set_pvalue`
    builds the set test."""
    adata = factors_adata()
    obs = cast(pd.DataFrame, adata.obs)
    cond = SPEC.condition_column
    spec = association(CONTRAST)
    # Condition order is np.unique's, as for the CP condition factor
    labels = prepare_group_labels(
        get_sample_to_group(adata, cond, spec["group_column"]),
        mapping=spec.get("mapping"),
    )
    case = (labels == CASE_LABEL).to_numpy()
    patients = (
        obs.drop_duplicates(cond).set_index(cond).loc[labels.index][SPEC.patient_column]
    )
    depth = np.bincount(
        obs["condition_unique_idxs"].astype(int),
        weights=np.asarray(cast(Any, adata.X).sum(axis=1)).ravel(),
    )

    D = load_factors_netcdf(SPEC.factors_path())["cp_D"]
    result = {}
    for k in range(1, D.shape[1] + 1):
        scores = component_lr_scores(DATASET, k) / depth[:, None]
        observed, null = patient_permutation_differential(
            rankdata(scores, axis=0), case, patients.astype(str).to_numpy(), N_PERMS
        )
        sd = null.std(axis=0)
        order = np.argsort(-np.abs(D.values[:, k - 1]))
        result[k] = {
            "z": observed / sd,
            "p": ((np.abs(null) >= np.abs(observed)).sum(0) + 1) / (N_PERMS + 1),
            "order": order,
            "medians": (np.median(scores[case], 0), np.median(scores[~case], 0)),
            "null_z": null[:, order[:MAX_TOP]] / sd[order[:MAX_TOP]],
        }
    return result


def set_statistic(z: np.ndarray, n_top: int) -> np.ndarray:
    """Mean |standardized difference| of the first `n_top` pairs (last axis)."""
    return np.abs(z[..., :n_top]).mean(axis=-1)


def set_pvalue(component: dict, n_top: int) -> float:
    """Fraction of permutations whose top-`n_top` set statistic reaches the
    observed one."""
    observed = set_statistic(component["z"][component["order"]], n_top)
    null = set_statistic(component["null_z"], n_top)
    return ((null >= observed).sum() + 1) / (len(null) + 1)


def makeFigure():
    return common.standalone_fixed(__name__)
