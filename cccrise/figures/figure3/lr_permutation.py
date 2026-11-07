"""Patient-permutation test of whether the outcome-associated CP component's
top LR pairs are larger in CLAD/Died than in stable/recovered samples.

Each LR pair is scored per condition in each component's own
sender/receiver context (`coordinated_changes.component_lr_scores`), divided
by the condition's read depth (as the plotted condition loadings are). The
CLAD/Died minus stable/recovered mean score difference is compared against `N_PERMS`
permutations of the outcome labels by patient
(`patient_permutation_differential`); a pair's p is the one-sided fraction
of permutations reaching its observed difference.

The panel is each of the component's `N_TOP` most heavily weighted pairs'
permutation null beside its observed difference, both in units of the
null's standard deviation so the pairs share an axis, colored by the
pair's Benjamini-Hochberg q across the `N_TOP` pairs. `lr_volcano` shows
the top pairs of every component.
"""

from functools import cache
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import false_discovery_control

from ...anova import get_sample_to_group, prepare_group_labels
from ...factors import load_factors_netcdf
from ...validation.significance import patient_permutation_differential
from .. import common, style
from ._common import COMPONENT, DATASET, SPEC, factors_adata
from .coordinated_changes import N_TOP, component_lr_scores

CASE_LABEL = "CLAD/Died"
# 6-month outcome collapsed to CLAD/Died vs. Stable/Recovered (no CLAD)
CONTRAST_COLUMN = "6monthcondition"
CONTRAST_MAPPING = {
    "CLAD": CASE_LABEL,
    "Died": CASE_LABEL,
    "No CLAD/Stable": "Stable/Recovered",
    "No CLAD/Recovered": "Stable/Recovered",
}
# The groups as axis labels name them
CASE_DISPLAY = "CLAD/died"
CONTROL_DISPLAY = "stable/recovered"
N_PERMS = 10_000
AXES = ("square",)


def plot(axs):
    component = permutation_analysis()[COMPONENT]
    plot_pair_nulls(component, N_TOP, axs[0])
    top = component["order"][:N_TOP]
    p = component["p"][top]
    common.source_data(
        "top_pair_permutation_tests",
        pd.DataFrame(
            {
                "LR pair": LR_NAMES[top],
                "Observed difference": component["observed"][top],
                "Null SD": component["null"].std(axis=0),
                "Permutation p": p,
                "BH q": false_discovery_control(p),
            }
        ),
        f"The CP component {COMPONENT}'s {N_TOP} most heavily weighted LR pairs: "
        f"{CASE_DISPLAY} minus {CONTROL_DISPLAY} mean depth-normalized score, "
        f"the SD of its null over {N_PERMS} patient-block permutations of the "
        "outcome (the plot's units), one-sided permutation p, and p "
        f"Benjamini-Hochberg adjusted across the {N_TOP} pairs.",
    )
    common.source_data(
        "top_pair_permutation_null",
        pd.DataFrame(component["null"], columns=LR_NAMES[top])
        .rename_axis(index="Permutation", columns="LR pair")
        .stack()
        .rename("Null difference")
        .reset_index(),
        "Each permutation's group difference for every top pair (seeded).",
    )
    common.source_data(
        "top_pair_scores_by_condition",
        pd.DataFrame(component["scores"][:, top], columns=LR_NAMES[top])
        .assign(
            Condition=component["conditions"],
            Patient=component["patients"],
            Group=np.where(component["case"], CASE_LABEL, "Stable/Recovered"),
        )
        .melt(
            id_vars=["Condition", "Patient", "Group"],
            var_name="LR pair",
            value_name="Depth-normalized score",
        ),
        "Each top pair's per-condition score in the component's sender/receiver "
        "context, divided by the condition's read depth, with the condition's "
        "patient (the permutation unit) and outcome group.",
    )


def plot_pair_nulls(component: dict, n_top: int, ax) -> None:
    """One row per top pair: its permutation null (violin) and observed
    difference (diamond), standardized by the null's standard deviation,
    colored by its BH q across the `n_top` pairs."""
    top = component["order"][:n_top]
    null = component["null"][:, :n_top]
    sd = null.std(axis=0)
    rows = np.arange(n_top)[::-1]
    bodies = ax.violinplot(
        null / sd, positions=rows, vert=False, showextrema=False, widths=0.8
    )
    for body in bodies["bodies"]:
        body.set(facecolor=style.COLORS["muted"], edgecolor="none", alpha=0.8)
    q = false_discovery_control(component["p"][top])
    ax.scatter(
        component["observed"][top] / sd,
        rows,
        marker="D",
        s=style.POINT_SIZE,
        color=np.where(
            q < common.SIG_ALPHA, style.COLORS["emphasis"], style.COLORS["ink"]
        ),
        zorder=3,
    )
    common.reference_line(ax, 0, horizontal=False, linestyle=":")
    ax.set_yticks(rows, LR_NAMES[top], fontsize=common.FACTOR_LABEL_FONT_SIZE)
    ax.set_ylim(-0.6, n_top - 0.4)
    ax.set_xlabel(f"Standardized mean score\n{CASE_DISPLAY} − {CONTROL_DISPLAY}")
    common.legend(
        ax,
        "above",
        ncols=2,
        handles=[
            Patch(color=style.COLORS["muted"], alpha=0.8, label="Patient-permuted"),
            *(
                Line2D(
                    [],
                    [],
                    marker="D",
                    linestyle="",
                    color=style.COLORS[color],
                    label=f"Observed, q {sign} {common.SIG_ALPHA}",
                )
                for color, sign in (("emphasis", "<"), ("ink", "≥"))
            ),
        ],
        fontsize=style.FONT_SIZE_DENSE,
        columnspacing=0.8,
    )


@cache
def permutation_analysis() -> dict[int, dict]:
    """Per component, from its own context scores: every LR pair's observed
    difference (`observed`) and one-sided permutation p (`p`), the pairs
    ordered by |loading| (`order`), each group's mean score (`means`), and
    the null differences of its `N_TOP` heaviest pairs (`null`,
    permutations x pairs, in `order`)."""
    adata = factors_adata()
    obs = cast(pd.DataFrame, adata.obs)
    cond = SPEC.condition_column
    # Condition order is np.unique's, as for the CP condition factor
    labels = prepare_group_labels(
        get_sample_to_group(adata, cond, CONTRAST_COLUMN), mapping=CONTRAST_MAPPING
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
            scores, case, patients.astype(str).to_numpy(), N_PERMS
        )
        order = np.argsort(-np.abs(D.values[:, k - 1]))
        result[k] = {
            "scores": scores,
            "conditions": labels.index.to_numpy(),
            "patients": patients.astype(str).to_numpy(),
            "case": case,
            "observed": observed,
            "p": ((null >= observed).sum(0) + 1) / (N_PERMS + 1),
            "order": order,
            "means": (scores[case].mean(0), scores[~case].mean(0)),
            "null": null[:, order[:N_TOP]],
        }
    return result


LR_NAMES = np.asarray(
    load_factors_netcdf(SPEC.factors_path())["lr_pair"].values
).astype(str)


def makeFigure():
    return common.standalone_fixed(__name__)
