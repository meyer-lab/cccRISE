"""Shared loading and labels for the NicheNet-vs-SecAct prior comparison
panels (`uv run cccrise-prior-comparison`)."""

from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ...runners.datasets import VALIDATION_DIR
from .. import common

DATASET = "alad"
SETTING = ["prior", "profile", "scoring"]
PRIOR_LABEL = {"nichenet": "NicheNet v2", "secact": "SecAct"}
PROFILE_LABEL = {"wilcoxon": "Wilcoxon", "log_tpm_diff": "pseudobulk"}
SCORING_LABEL = {"auroc": "AUROC", "ridge": "ridge z"}
#: Each prior's settings in this order; the first is this project's pipeline,
#: the last SecAct's.
SETTING_ORDER = [
    ("wilcoxon", "auroc"),
    ("wilcoxon", "ridge"),
    ("log_tpm_diff", "auroc"),
    ("log_tpm_diff", "ridge"),
]
NATIVE = {("wilcoxon", "auroc"): "ours", ("log_tpm_diff", "ridge"): "SecAct's"}


def load() -> pd.DataFrame:
    return common.load_runner_output(
        f"nichenet_prior_comparison_{DATASET}.parquet",
        VALIDATION_DIR,
        hint=f"    uv run cccrise-prior-comparison --dataset {DATASET}",
    )


def settings() -> list[tuple[str, str, str]]:
    return [(p, *s) for p in PRIOR_LABEL for s in SETTING_ORDER]


def setting_label(prior: str, profile: str, scoring: str) -> str:
    label = f"{PROFILE_LABEL[profile]} · {SCORING_LABEL[scoring]}"
    if (profile, scoring) in NATIVE:
        label += f" ({NATIVE[profile, scoring]})"
    return f"{PRIOR_LABEL[prior]}: {label}"


def within_component_ranks(scores: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return cast(pd.DataFrame, scores.groupby("component")[columns].rank(pct=True))


def weight_score_correlation(scores: pd.DataFrame, control: str | None = None) -> float:
    """Spearman of ligand weight vs score, pooled over components after
    ranking within each; with `control`, both are first residualized on it."""
    cols = ["weight", "score"] + ([control] if control else [])
    r = within_component_ranks(scores.dropna(subset=cols), cols)
    if control:
        r = r[["weight", "score"]].apply(
            lambda y: y - np.polyval(np.polyfit(r[control], y, 1), r[control])
        )
    return float(spearmanr(r["weight"], r["score"])[0])
