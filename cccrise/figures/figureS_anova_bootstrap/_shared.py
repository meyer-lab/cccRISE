"""The patient-resampled ANOVA bootstrap shared by this figure's panels,
from ``uv run cccrise-anovaboot``.

Each replicate refits RISE and CP on a resampled patient roster, matches its
components back onto the full-cohort reference, and records their ANOVA
statistics, so every component's effect size comes with a distribution over
rosters. Components keep the reference's numbering.
"""

from functools import cache

import numpy as np
import pandas as pd

from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...runners.provenance import read_metadata
from ...validation.anova_stability import summarize_bootstrap_anova
from .. import common, style

DATASET = "alad"
FILENAME = DATASETS[DATASET].anova_bootstrap_filename()


@cache
def metadata() -> dict:
    return read_metadata(VALIDATION_DIR / FILENAME)


@cache
def replicates() -> pd.DataFrame:
    return common.load_runner_output(FILENAME, VALIDATION_DIR)


@cache
def summary() -> pd.DataFrame:
    """Per component: bootstrap interval, significance / sign stability, the
    full-cohort reference statistics, and the reference's top LR pair."""
    meta = metadata()
    n = len(meta["reference_eta_squared"])
    reference = pd.DataFrame(
        {
            "Component": np.arange(1, n + 1),
            "EtaSquared": meta["reference_eta_squared"],
            "F": meta.get("reference_f", [np.nan] * n),
            "P": meta["reference_p"],
            "GroupMeanDiff": meta.get("reference_group_mean_diff", [np.nan] * n),
        }
    )
    table = summarize_bootstrap_anova(
        replicates(), reference, ci_level=meta["ci_level"], alpha=alpha()
    )
    table["TopLRPair"] = meta["reference_top_lr_pair"]
    return table


def alpha() -> float:
    return float(metadata()["alpha"])


def reference_colors(table: pd.DataFrame) -> dict[int, str]:
    """Emphasis for components significant in the full cohort, muted otherwise."""
    return {
        int(c): style.COLORS["emphasis"] if p < alpha() else style.COLORS["muted"]
        for c, p in zip(table["Component"], table["ReferenceP"], strict=True)
    }


def outcome() -> str:
    return metadata()["group_column"]
