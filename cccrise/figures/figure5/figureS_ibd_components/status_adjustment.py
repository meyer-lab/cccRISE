"""Every CP component's inflammation-status effect, alone and adjusted:
partial eta-squared (type II) of status in OLS of the rank of the
component's read-depth-corrected loading across biopsies (ranks, so the few
biopsies carrying most of C1's and C5's loading do not dominate), alone,
with its carrier cell types' and the immune lineages' fractions of the
biopsy (`CARRIERS`, `IMMUNE`), and with the biopsy's dissociation score
(`run_ibd_programs.DISSOCIATION_GENES`); * status p < `common.SIG_ALPHA`."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from ... import common, style
from .. import _common as S

AXES = ("half",)
#: Each component's sender and receiver carrier fine types (its heaviest
#: mean weights, `celltype_weights`)
CARRIERS = {
    1: ["Monocytes SOD2+"],
    2: ["Mast cells", "ILCs"],
    3: ["Enteroendocrine cells", "Enterochromaffin cells"],
    4: ["Enteroendocrine cells", "Tuft cells"],
    5: ["Monocytes SOD2+"],
    6: ["ILCs", "Enterochromaffin cells"],
}
IMMUNE = ["B cell", "Plasma B cell", "T cell", "Myeloid"]
FLOOR = 0.5  # cells added to every count before its log fraction
STATUS = "C(status, Treatment('Healthy'))"
MODELS = {
    "Status alone": ("", style.COLORS["muted"]),
    "+ composition": ("composition", style.COLORS["emphasis"]),
    "+ dissociation": ("dissociation", style.CATEGORICAL[1]),
}


def biopsy_table() -> pd.DataFrame:
    """Per biopsy: status, each component's loading rank, its carriers' log
    fractions, the immune fraction and the dissociation score."""
    samples = S.samples().set_index(S.SPEC.condition_column)
    counts = S.celltype_counts()
    total = counts.groupby(S.SPEC.condition_column)["n_cells"].sum()
    fine = counts.pivot_table(
        index=S.SPEC.condition_column,
        columns="celltype",
        values="n_cells",
        fill_value=0,
    )
    lineage = counts.pivot_table(
        index=S.SPEC.condition_column,
        columns="lineage",
        values="n_cells",
        aggfunc="sum",
        fill_value=0,
    )
    loadings = S.corrected_loadings("cp_A").set_axis(samples.index)
    table = pd.DataFrame(
        {
            "status": samples[S.GROUP_COLUMN],
            "immune": lineage[IMMUNE].sum(axis=1) / total,
            "dissociation": S.programs("dissociation").set_index("Biopsy")[
                "Dissociation score"
            ],
        }
    )
    for k, types in CARRIERS.items():
        table[f"rank_C{k}"] = loadings[k].rank()
        for i, celltype in enumerate(types):
            table[f"carrier_C{k}_{i}"] = np.log10((fine[celltype] + FLOOR) / total)
    return table


def status_effects(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in CARRIERS:
        covariates = {
            "": "",
            "composition": " + "
            + " + ".join([*table.filter(like=f"carrier_C{k}_").columns, "immune"]),
            "dissociation": " + dissociation",
        }
        for model, (extra, _) in MODELS.items():
            anova = anova_lm(
                smf.ols(f"rank_C{k} ~ {STATUS}{covariates[extra]}", data=table).fit(),
                typ=2,
            )
            ss = anova["sum_sq"]
            rows.append(
                {
                    "Component": k,
                    "Model": model,
                    "Status partial eta2": ss[STATUS] / (ss[STATUS] + ss["Residual"]),
                    "Status p": anova.loc[STATUS, "PR(>F)"],
                }
            )
    return pd.DataFrame(rows)


def plot(axs):
    (ax,) = axs
    table = biopsy_table()
    effects = status_effects(table)
    common.source_data(
        "status_effect_adjusted",
        effects,
        "Partial eta-squared and p (type II) of inflammation status in OLS of each "
        "component's read-depth-corrected loading rank across biopsies, alone, "
        "with its carriers' log fractions and the immune fraction, and with the "
        "biopsy's mean dissociation score.",
    )
    for i, (model, (_, color)) in enumerate(MODELS.items()):
        rows = effects[effects["Model"] == model]
        x = rows["Component"].to_numpy() + 0.27 * (i - 1)
        ax.bar(x, rows["Status partial eta2"], width=0.26, color=color, label=model)
        for xi, eta, p in zip(
            x, rows["Status partial eta2"], rows["Status p"], strict=True
        ):
            if p < common.SIG_ALPHA:
                ax.text(
                    xi,
                    eta,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=style.FONT_SIZE_DENSE,
                )
    ax.set_xticks(list(CARRIERS))
    ax.set_xlabel("CP component")
    ax.set_ylabel("Status partial $\\eta^2$\n(loading rank)")
    common.legend(ax, "right", fontsize=style.FONT_SIZE_DENSE)


def makeFigure():
    return common.standalone_fixed(__name__)
