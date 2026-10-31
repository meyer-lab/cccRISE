"""Association of each biopsy covariate with every inflammation-status
association tested against the components (one-way ANOVA eta-squared,
uncorrected p), to flag covariates that could confound a
component-inflammation association. A binary covariate is coded 0/1, so its
eta-squared is the squared Cramér's V; biopsies missing a covariate are left
out of its tests."""

from functools import cache

import numpy as np
import pandas as pd

from ... import common
from ...figure3.figureS_alad_associations.covariates import covariate_anova, draw
from .. import _common as S
from ._common import save_association_data

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    table = covariate_table()
    draw(ax, table)
    save_association_data(
        table,
        covariates(),
        "covariate",
        "each biopsy covariate (binary covariates coded 0/1, blank where not "
        "recorded; age in 10-year bins; cells after the 2,000-cell cap; read "
        "depth as the total normalized expression the loadings are corrected for)",
        "P is the raw, uncorrected p-value (stars mark P < 0.05)",
    )


@cache
def covariates() -> pd.DataFrame:
    """Each biopsy's covariates, in `S.samples` order."""
    samples = S.samples()
    chip = samples["chip_version"].astype(object)
    return pd.DataFrame(
        {
            "Age": samples["age_binned_10"],
            "Male": samples["sex"] == "male",
            "Chip B": np.where(chip.isna(), np.nan, chip == "chip-b"),
            "Cells": samples["n_cells"],
            "Read depth": samples["read_depth"],
        }
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
