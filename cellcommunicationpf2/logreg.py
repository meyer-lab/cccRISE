import anndata
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold

from .utils import add_obs_cmp_label


def ccc_rise_logreg_weights(
    X,
    sample_to_group,
    scoring=None,
    random_state=0,
):
    """
    Evaluate logistic regression weights using CPD factors.
    """
    if scoring is None:
        scoring = ["roc_auc", "accuracy"]

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=random_state)
    for i in scoring:
        lr_fit = LogisticRegressionCV(
            random_state=random_state,
            max_iter=5000,
            solver="saga",
            l1_ratios=(1.0,),
            cv=cv,
            scoring=i,
            use_legacy_attributes=False,
        ).fit(X.uns["A"], sample_to_group)
        if i == "roc_auc":
            weights_aucroc = lr_fit.coef_[0]
        elif i == "accuracy":
            weights_accuracy = lr_fit.coef_[0]

    return weights_aucroc, weights_accuracy


def population_gene_logreg_weights(
    X: anndata.AnnData,
    cmp: int,
    factor_type: str,
    pos: bool = True,
    top_perc: float = 10,
    random_state: int = 0,
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
