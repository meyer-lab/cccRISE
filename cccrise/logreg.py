import anndata
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold

from .figures.commonFuncs.plotGeneral import binarize_group, get_sample_to_group
from .utils import add_obs_cmp_label, correct_conditions

#: Scorings `ccc_rise_logreg_weights` reports, in the order it returns them.
CCC_RISE_SCORINGS = ("roc_auc", "accuracy")


def fit_logreg_weights(
    X: anndata.AnnData,
    sample_to_group: pd.Series,
    scoring: str = "roc_auc",
    random_state: int | None = 0,
) -> np.ndarray:
    """Fit one cross-validated L1 logistic regression on the condition factor
    matrix `X.uns["A"]` and return its per-component weights.

    The single-scoring unit `ccc_rise_logreg_weights` is built from: an
    analysis that only needs one scoring (a bootstrap, say, where the fit is
    repeated hundreds of times) pays for one fit rather than two.

    Parameters
    ----------
    X : anndata.AnnData
        Carries the condition factor matrix in `uns["A"]`, one row per
        condition in `condition_unique_idxs` order.
    sample_to_group : pd.Series
        Binary class label per condition, aligned to `uns["A"]`'s rows.
    scoring : str, default="roc_auc"
        Scikit-learn scoring used to select the regularization strength.
    random_state : int | None, default=0
        Seed for the cross-validation splits and the solver.

    Returns
    -------
    np.ndarray
        One weight per CP component.
    """
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=random_state)
    lr_fit = LogisticRegressionCV(
        random_state=random_state,
        max_iter=10000,
        solver="saga",
        l1_ratios=(1.0,),
        cv=cv,
        scoring=scoring,
        use_legacy_attributes=False,
    ).fit(X.uns["A"], sample_to_group)
    return lr_fit.coef_[0]


def ccc_rise_logreg_weights(
    X: anndata.AnnData,
    sample_to_group: pd.Series,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate logistic regression weights using CPD factors, selecting the
    regularization strength by each of `CCC_RISE_SCORINGS` in turn.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Per-component weights from the AUC-ROC-selected and the
        accuracy-selected fit, in that order.
    """
    weights_aucroc, weights_accuracy = (
        fit_logreg_weights(
            X, sample_to_group, scoring=scoring, random_state=random_state
        )
        for scoring in CCC_RISE_SCORINGS
    )
    return weights_aucroc, weights_accuracy


def prepare_logreg_inputs(
    adata: anndata.AnnData,
    factors: list[np.ndarray],
    condition_column: str,
    group_column: str,
    control_label: str,
    other_label: str,
) -> tuple[anndata.AnnData, pd.Series]:
    """Attach one CP fit's factors to an AnnData object and derive the binary
    class label the logistic regression is fit against.

    Mirrors the condition-factor read-depth correction and group labeling
    `figures/figure4.py` applies to the primary post hoc factorization, so
    weights computed from any other fit -- a different rank, a bootstrap
    replicate -- are directly comparable to the ones reported there.

    Parameters
    ----------
    adata : anndata.AnnData
        Data the fit came from, carrying `condition_unique_idxs` in `obs`.
    factors : list[np.ndarray]
        Standardized CP factors [A, B, C, D].
    condition_column, group_column : str
        Columns in `obs` holding the condition identity and its group label.
    control_label : str
        Value of `group_column` mapped to the reference class.
    other_label : str
        Label applied to every other value of `group_column`.

    Returns
    -------
    tuple[anndata.AnnData, pd.Series]
        A copy carrying the factors in `uns` with `uns["A"]` read-depth
        corrected, and the binary class label per condition.
    """
    X = adata.copy()
    X.uns["A"], X.uns["B"], X.uns["C"], X.uns["D"] = factors
    X.uns["A"] = correct_conditions(X)

    sample_to_group = get_sample_to_group(X, condition_column, group_column)
    return X, binarize_group(sample_to_group, control_label, control_label, other_label)


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
