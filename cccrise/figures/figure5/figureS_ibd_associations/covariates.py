"""Association of each biopsy covariate with every inflammation-status
association tested against the components (one-way ANOVA eta-squared,
uncorrected p), to flag covariates that could confound a
component-inflammation association, as ALAD's are screened. A binary
covariate is coded 0/1, so its eta-squared is the squared Cramér's V;
biopsies missing a covariate are left out of its tests. Each lineage's
fraction of the biopsy's cells follows the sample covariates."""

from functools import cache

import numpy as np
import pandas as pd

from ... import common
from ...figure3.figureS_alad_associations.covariates import covariate_anova, draw
from .. import _common as S
from ..figureS_ibd_cohort._common import ordered_celltypes, proportions
from ._common import save_association_data

AXES = ("wide",)


def plot(axs):
    (ax,) = axs
    table = covariate_table()
    draw(ax, table)
    save_association_data(
        table,
        covariates(),
        "covariate",
        "each biopsy covariate (binary covariates coded 0/1, blank where not "
        "recorded; age in 10-year bins; read depth as the total normalized "
        "expression the loadings are corrected for; each lineage as its "
        "fraction of the biopsy's cells after the 2,000-cell cap)",
    )


@cache
def covariates() -> pd.DataFrame:
    """Each biopsy's covariates, in `S.samples` order: sample metadata, then
    each lineage's fraction of its cells."""
    samples = S.samples()
    chip = samples["chip_version"].astype(object)
    lineages = proportions("lineage").reindex(samples[S.SPEC.condition_column])
    return pd.DataFrame(
        {
            "Age": samples["age_binned_10"],
            "Male": samples["sex"] == "male",
            "Chip B": np.where(chip.isna(), np.nan, chip == "chip-b"),
            "Read depth": samples["read_depth"],
        }
        | {name: lineages[name].to_numpy() for name in ordered_celltypes("lineage")}
    ).astype(float)


@cache
def covariate_table() -> pd.DataFrame:
    """Each covariate's uncorrected one-way ANOVA against every IBD
    association (`covariate_anova`): a confounder screen errs toward
    flagging."""
    values = covariates()
    return covariate_anova(
        S.condition_adata(values), values, S.ASSOCIATIONS, S.SPEC.condition_column
    )


def makeFigure():
    return common.standalone_fixed(__name__)
