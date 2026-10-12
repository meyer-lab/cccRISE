"""cccRISE's own patient-resampled ANOVA bootstrap on ALAD, which the
comparator bootstrap panel sets beside the comparators' (`bootstrap`).

`main` replaced that bootstrap with a jackknife (#223) and dropped its
figure; its ALAD output (`anova_bootstrap_alad.parquet`, from the former
``cccrise-anovaboot``) is still in the repo and is read here as it was."""

from functools import cache

import numpy as np
import pandas as pd

from ...comparators.resampling import summarize_bootstrap_anova
from ...runners.datasets import VALIDATION_DIR
from ...runners.provenance import read_metadata
from .. import common

FILENAME = "anova_bootstrap_alad.parquet"


@cache
def metadata() -> dict:
    return read_metadata(VALIDATION_DIR / FILENAME)


@cache
def summary() -> pd.DataFrame:
    """Per component: bootstrap interval, significance / sign stability and
    the full-cohort reference statistics."""
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
        common.load_runner_output(FILENAME, VALIDATION_DIR),
        reference,
        ci_level=meta["ci_level"],
        alpha=float(meta["alpha"]),
    )
    table["TopLRPair"] = meta["reference_top_lr_pair"]
    return table
