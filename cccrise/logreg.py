"""Cell-level logistic regression of gene expression against a component's
sender/receiver association.

The component--phenotype half of this module moved to `cccrise.anova`: a
per-component ANOVA reports an F, a p and an effect size without a
regularization path, so a bootstrap over the patient roster measures sampling
variability rather than L1 selection. What is left here is a genuine logistic
regression at the *cell* level, which the ANOVA does not replace.
"""

import anndata
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .utils import add_obs_cmp_label


def population_gene_logreg_weights(
    X: anndata.AnnData,
    cmp: int,
    factor_type: str,
    pos: bool = True,
    top_perc: float = 10,
    random_state: int = 42,
) -> pd.Series:
    """
    Fit an L1 logistic regression on per-cell gene expression to distinguish the
    top (or bottom) `top_perc` percent of a population by its sender/receiver
    association with `cmp` from the remainder of that population. Returns the
    fitted per-gene weights, sorted from most to least positive.
    """
    X = add_obs_cmp_label(
        X.copy(), cmp=cmp, pos=pos, top_perc=top_perc, type=factor_type
    )
    y = X.obs[f"Cmp{cmp}"].to_numpy()

    expr = X.X
    if hasattr(expr, "tocsr"):
        expr = expr.tocsr()

    lr_fit = LogisticRegression(
        solver="saga",
        l1_ratio=1.0,
        random_state=random_state,
        tol=1e-3,
        max_iter=1000,
    ).fit(expr, y)
    weights = pd.Series(lr_fit.coef_[0], index=X.var_names)
    return weights.sort_values(ascending=False)
