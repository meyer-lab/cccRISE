"""Data shared by Figure 3's panels: the ALAD outcome associations tested
against every CP component, and the fit they are tested on.

`ASSOCIATIONS` is also the Shiny app's list of ALAD comparisons.
"""

from functools import cache
from typing import Any

import anndata
import pandas as pd

from ...anova import _resolve_factors_and_metadata, run_multiple_factors_anova
from ...runners.datasets import DATASETS

DATASET = "alad"
SPEC = DATASETS[DATASET]

#: Outcome associations tested against every CP component: `name` labels the
#: row, `group_column` is the condition-level `obs` column, and the optional
#: `levels` / `mapping` / `control_label` / `other_label` regroup it (see
#: `anova.prepare_group_labels`).
ASSOCIATIONS: list[dict[str, Any]] = [
    # --- Primary ALAD Contrasts (Previously Tested) ---
    {
        "name": "Control vs. ALAD (All)",
        "group_column": "ALADstatus",
        "control_label": "control",
        "other_label": "alad",
    },
    {
        "name": "Control vs. Declined",
        "group_column": "ALADstatus",
        "levels": ["control", "declined"],
        "control_label": "control",
    },
    {
        "name": "Control vs. Recovered",
        "group_column": "ALADstatus",
        "levels": ["control", "recovered"],
        "control_label": "control",
    },
    {
        "name": "Declined vs. Recovered",
        "group_column": "ALADstatus",
        "levels": ["declined", "recovered"],
        "control_label": "declined",
    },
    {
        "name": "ALAD Status (3-Way)",
        "group_column": "ALADstatus",
    },
    # --- Recommended Longitudinal Graft Outcomes & Rejection ---
    {
        "name": "6-Month Outcome (4-Way)",
        "group_column": "6monthcondition",
    },
    {
        "name": "6-Month CLAD/Died vs. No CLAD",
        "group_column": "6monthcondition",
        "mapping": {
            "CLAD": "CLAD/Died",
            "Died": "CLAD/Died",
            "No CLAD/Stable": "No CLAD",
            "No CLAD/Recovered": "No CLAD",
        },
        "control_label": "No CLAD",
    },
    {
        "name": "1-Year CLAD/Died vs. No CLAD",
        "group_column": "1yearcondition",
        "mapping": {
            "No CLAD/Stable": "No CLAD",
            "No CLAD/Recovered": "No CLAD",
            "CLAD/Declined": "CLAD/Died",
            "CLAD/Stable": "CLAD/Died",
            "Died": "CLAD/Died",
            "No CLAD/Declined": "No CLAD",
            "Recovered": "No CLAD",
        },
        "control_label": "No CLAD",
    },
    {
        "name": "Acute Rejection (Yes vs. No)",
        "group_column": "rejection.y",
        "levels": ["no", "yes"],
        "control_label": "no",
    },
    {
        "name": "Recovery Trajectory (5-Way)",
        "group_column": "recovered.y",
    },
    # --- Previously Tested Demographics & Baseline Covariates ---
    {
        "name": "Pre-Tx Diagnosis (ILD vs. Other)",
        "group_column": "diagnosisgroup",
        "mapping": {
            "ILD": "ILD",
            "pHTN": "Other",
            "COPD": "Other",
            "CF": "Other",
        },
        "control_label": "ILD",
    },
    {
        "name": "CMV Serostatus (IgG+ vs. IgG-)",
        "group_column": "cmvstatus2",
        "levels": ["IgG+", "IgG-"],
        "control_label": "IgG-",
    },
    {
        "name": "Sex (Male vs. Female)",
        "group_column": "sex",
        "levels": ["male", "female"],
        "control_label": "female",
    },
]


@cache
def factors_adata() -> anndata.AnnData:
    """The canonical fit attached to the ALAD source data's annotations."""
    adata, _, _ = _resolve_factors_and_metadata(
        SPEC.factors_path(), condition_column=SPEC.condition_column
    )
    return adata


@cache
def association_table() -> pd.DataFrame:
    """ANOVA of every CP component against every association, FDR-corrected
    across components within each association."""
    return run_multiple_factors_anova(
        factors_adata(),
        ASSOCIATIONS,
        condition_column=SPEC.condition_column,
        correction="fdr_bh",
    )


def association(name: str) -> dict[str, Any]:
    return next(spec for spec in ASSOCIATIONS if spec["name"] == name)
