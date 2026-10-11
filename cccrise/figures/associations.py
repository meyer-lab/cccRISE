"""The condition-level associations each dataset's CP and RISE components are
tested against (`anova.run_multiple_factors_anova`), which are also the Shiny
app's comparison lists.

Each entry's `name` labels the row, `group_column` is the condition-level
`obs` column, and the optional `levels` / `mapping` / `control_label` /
`other_label` regroup it (see `anova.prepare_group_labels`).
"""

from typing import Any

ALAD_ASSOCIATIONS: list[dict[str, Any]] = [
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

#: IBD's inflammation-status contrasts, then the demographics IBDverse records
#: per sample (each sample is its own individual).
IBD_ASSOCIATIONS: list[dict[str, Any]] = [
    {
        "name": "Healthy vs. Crohn's (All)",
        "group_column": "inflammation_status",
        "control_label": "Healthy",
        "other_label": "Crohn's",
    },
    {
        "name": "Healthy vs. CD Inflamed",
        "group_column": "inflammation_status",
        "levels": ["Healthy", "CD_inflamed"],
        "control_label": "Healthy",
    },
    {
        "name": "Healthy vs. CD Uninflamed",
        "group_column": "inflammation_status",
        "levels": ["Healthy", "CD_uninflamed"],
        "control_label": "Healthy",
    },
    {
        "name": "CD Uninflamed vs. CD Inflamed",
        "group_column": "inflammation_status",
        "levels": ["CD_uninflamed", "CD_inflamed"],
        "control_label": "CD_uninflamed",
    },
    {
        "name": "Inflammation Status (3-Way)",
        "group_column": "inflammation_status",
    },
    {
        "name": "Sex (Male vs. Female)",
        "group_column": "sex",
        "levels": ["male", "female"],
        "control_label": "female",
    },
    {
        "name": "Age (10-Year Bins)",
        "group_column": "age_binned_10",
    },
]

ASSOCIATIONS: dict[str, list[dict[str, Any]]] = {
    "alad": ALAD_ASSOCIATIONS,
    "ibd": IBD_ASSOCIATIONS,
}
