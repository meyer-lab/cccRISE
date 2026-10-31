"""Source data shared by this figure's association heatmaps: the ANOVA table
each plots, and the per-biopsy values it tested beside each biopsy's group
in every association."""

import pandas as pd

from ....anova import prepare_group_labels
from ... import common
from ...figure3.figureS_alad_associations._common import TABLE_COLUMNS
from ...figure3.outcome_association import REGROUPING_KEYS
from .. import _common as S


def save_association_data(
    table: pd.DataFrame, values: pd.DataFrame, variable: str, tested: str, p_values: str
) -> None:
    """Export the one-way ANOVA `table` of each `variable` (its 'Component'
    column) against each of `S.ASSOCIATIONS`, and the per-biopsy `values`
    (biopsies x variables, in `S.samples` order) it tested, `tested`
    describing them; `p_values` describes the P and PAdj columns."""
    common.source_data(
        f"anova_{variable}_inflammation",
        table[[c for c in TABLE_COLUMNS if c in table]].rename(
            columns={"Component": variable.capitalize()}
        ),
        f"One row per (inflammation association, {variable}): one-way ANOVA of "
        f"the {variable}'s per-biopsy value across the association's groups. "
        f"EtaSquared is the effect size plotted; {p_values}. GroupSizes lists "
        "the biopsies per group.",
    )
    groups = association_groups()
    common.source_data(
        f"{variable}_values_by_biopsy",
        pd.concat([groups, values.add_prefix(f"{variable.capitalize()} ")], axis=1),
        "One row per terminal ileum biopsy: its group in each inflammation "
        f"association (blank where left out of that test) and {tested}.",
    )


def association_groups() -> pd.DataFrame:
    """Each biopsy (`S.samples` order) and its group in every association
    (blank where left out of it)."""
    status = S.samples().set_index(S.SPEC.condition_column)[S.GROUP_COLUMN]
    groups = {
        f"Group: {spec['name']}": prepare_group_labels(
            status, **{k: spec[k] for k in REGROUPING_KEYS if k in spec}
        ).to_numpy()
        for spec in S.ASSOCIATIONS
    }
    return pd.DataFrame({"Biopsy": status.index.to_numpy()} | groups)
