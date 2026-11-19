"""Eta-squared of one-way ANOVAs of each patient covariate against each ALAD
outcome association, with uncorrected p.

Association of each patient covariate with every ALAD outcome tested
against the components (one-way ANOVA eta-squared, uncorrected p), to flag
covariates that could confound a component-outcome association. A binary
covariate is coded 0/1, so its eta-squared is the squared Cramér's V."""

from functools import cache

import anndata
import pandas as pd

from ....anova import run_multiple_factors_anova
from ... import common
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
        "each patient covariate (binary covariates coded 0/1; age in years, "
        "baseline FEV1 as recorded)",
    )


def draw(ax, table: pd.DataFrame) -> None:
    """The covariate x association eta-squared heatmap, starred at
    uncorrected p < 0.05."""
    common.plot_association_heatmap(
        table, ax, xlabel="Covariate", p_column="P", p_label="p"
    )
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set(ha="right", rotation_mode="anchor")


@cache
def covariates() -> pd.DataFrame:
    """Each condition's covariates, in `S.condition_obs` order."""
    obs = S.condition_obs()
    return pd.DataFrame(
        {
            "Age": obs["age"],
            "Male": obs["sex"] == "male",
            "Caucasian": obs["ethnicity"] == "Caucasian",
            "ILD diagnosis": obs["diagnosisgroup"] == "ILD",
            "CMV IgG+": obs["cmvstatus2"] == "IgG+",
            "Baseline FEV1": obs["baselineFEV1"],
        }
    ).astype(float)


@cache
def covariate_table() -> pd.DataFrame:
    """`run_multiple_factors_anova` of each covariate, in place of a
    component, against each of `S.ASSOCIATIONS`, uncorrected: a confounder
    screen should err toward flagging."""
    values = covariates()
    adata = anndata.AnnData(obs=S.condition_obs(), uns={"A": values.to_numpy()})
    return covariate_anova(adata, values, S.ASSOCIATIONS, S.SPEC.condition_column)


def covariate_anova(
    adata: anndata.AnnData,
    values: pd.DataFrame,
    associations: list,
    condition_column: str,
) -> pd.DataFrame:
    """`run_multiple_factors_anova` of each column of `values` (carried as
    `adata`'s `uns["A"]`), named in its 'Component' column, against each of
    `associations`, uncorrected."""
    table = run_multiple_factors_anova(
        adata,
        associations,
        condition_column=condition_column,
        correction=None,
        read_depth_correct=False,
    )
    table["Component"] = values.columns[table["Component"].astype(int) - 1]
    return table


def makeFigure():
    return common.standalone_fixed(__name__)
