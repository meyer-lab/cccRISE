"""Source data shared by this figure's association heatmaps: the ANOVA table
each plots, and the per-sample values it tested."""

import pandas as pd

from ....anova import get_sample_to_group, prepare_group_labels
from ... import common
from .. import _common as S
from ..outcome_association import REGROUPING_KEYS

TABLE_COLUMNS = [
    "Association",
    "GroupColumn",
    "Component",
    "NGroups",
    "GroupSizes",
    "F",
    "EtaSquared",
    "P",
    "PAdj",
    "GroupMeanDiff",
]


def save_association_data(
    table: pd.DataFrame,
    values,
    variable: str,
    tested: str,
    correction: str | None = None,
) -> None:
    """Export the one-way ANOVA `table` of each `variable` (its 'Component'
    column) against each of `S.ASSOCIATIONS`, and the per-sample `values`
    (samples x variables, in condition order) it tested, `tested` describing
    them, beside each sample's group in every association. `correction`
    describes 'PAdj', which is left out if None (an uncorrected table)."""
    columns = [c for c in TABLE_COLUMNS if c in table and (correction or c != "PAdj")]
    p_values = (
        f"P is the raw p-value and PAdj {correction}"
        if correction
        else "P is the raw, uncorrected p-value (stars mark P < 0.05)"
    )
    common.source_data(
        f"anova_{variable}_outcome",
        table[columns].rename(columns={"Component": variable.capitalize()}),
        f"One row per (outcome association, {variable}): one-way ANOVA of the "
        f"{variable}'s per-sample value across the association's groups. "
        f"EtaSquared is the effect size plotted; {p_values}. GroupSizes lists "
        "the samples per group.",
    )
    adata = S.factors_adata()
    groups = {
        f"Group: {spec['name']}": prepare_group_labels(
            get_sample_to_group(adata, S.SPEC.condition_column, spec["group_column"]),
            **{k: spec[k] for k in REGROUPING_KEYS if k in spec},
        ).to_numpy()
        for spec in S.ASSOCIATIONS
    }
    obs = S.condition_obs()
    samples = pd.DataFrame(
        {"Sample": obs.index, "Patient": obs[S.SPEC.patient_column].to_numpy()}
        | groups
        | {f"{variable.capitalize()} {c}": values[c].to_numpy() for c in values}
    )
    common.source_data(
        f"{variable}_values_by_sample",
        samples,
        f"One row per BAL sample: its group in each outcome association (blank "
        f"where left out of that test) and {tested}.",
    )
