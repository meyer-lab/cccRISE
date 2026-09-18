"""Cross-validated phenotype AUC, defined once for every benchmarked method.

Reviewer #1 asked for an ALAD-status AUC reported for CCC-RISE alongside the
published tensor methods. `#191` had already replaced the patient-level L1
logistic regression with the per-component ANOVA in `cccrise.anova`, which
reports an F, a p and an effect size but no AUC -- the right call for the
stability work (an ANOVA has no regularization path, so a bootstrap over the
patient roster measures sampling variability alone) and the wrong one for a
comparator table, where a method with three components, a method with eight and
a method with a Tucker core all have to produce the same number.

So this module is not a return to the pre-#191 statistic. The ANOVA stays
primary in the manuscript; this is an additional, benchmark-only number whose
entire job is comparability.

What makes it comparable
------------------------
Every method on the tensor axis is reduced to a `(n_conditions, n_components)`
score matrix before it gets here, and `cv_auc` never learns which method
produced one. Whatever a method does to earn its scores, it is scored the same
way afterwards.

The classifier is pluggable (`CLASSIFIERS`) because the choice is a judgement
call and swapping it should not mean touching anything else: pass
`classifier="l2_logreg"` and every method in the table moves together. The
default is L1 logistic regression with the regularization strength selected
*inside* each training fold, which is the pre-#191 classifier
(`logreg.fit_logreg_weights` as it stood at `9453180`) and so keeps the new
number on the same footing as the one the manuscript used to report.

Nesting, and why it matters here
--------------------------------
Reviewer #1's comment #4 is that rank selection performed against the predictive
objective, outside the folds, biases the reported AUC upward. It does not apply
to the ranks: RISE 19 / CP 3 were selected by BiCV and CORCONDIA, never against
ALAD status. It does apply to the regularization strength, which *is* fit
against the objective -- so `C` is selected by an inner cross-validation on each
outer training fold and never sees the outer test fold.

The reported statistic
----------------------
`AUCResult.auc` is the mean over repeats of the **pooled out-of-fold AUC**: within
one repeat, every condition gets exactly one prediction from the fold that held
it out, and the AUC is computed once over all of them. The obvious alternative --
averaging the AUCs of the individual folds -- is much noisier on a cohort this
size (47 patients over 10 folds is ~5 held out at a time) and is undefined
whenever a fold happens to be single-class. Fold-level AUCs are still returned,
as a diagnostic, not as the headline.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

#: Anything that indexes one label (or group) per condition.
LabelLike = Sequence[Any] | pd.Series | np.ndarray

#: Grid of inverse regularization strengths the inner CV selects `C` from.
DEFAULT_CS = np.logspace(-4, 4, 20)

#: Signature every entry of `CLASSIFIERS` satisfies: given the inner-CV fold
#: count and a seed, return an unfitted estimator. The scaler belongs inside the
#: returned pipeline, never outside it -- scaling on the full matrix before
#: splitting leaks the test fold's scale into training.
ClassifierFactory = Callable[[int, int | None], Pipeline]


def _l1_logreg(inner_splits: int, random_state: int | None) -> Pipeline:
    """L1 logistic regression, `C` chosen by inner CV. The default."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=DEFAULT_CS,
            # sklearn >=1.8 deprecates `penalty` in favour of `l1_ratios`;
            # (1.0,) is L1, (0.0,) is L2. Passing `penalty="l1"` here would be
            # equivalent today and a FutureWarning on every fit.
            l1_ratios=(1.0,),
            solver="saga",
            cv=inner_splits,
            scoring="roc_auc",
            max_iter=10000,
            random_state=random_state,
            use_legacy_attributes=False,
        ),
    )


def _l2_logreg(inner_splits: int, random_state: int | None) -> Pipeline:
    """L2 logistic regression, `C` chosen by inner CV.

    The sensitivity row for the D1 decision: L1 selects among correlated
    components somewhat arbitrarily, which is exactly the behaviour that made
    the pre-#191 bootstrap hard to read. If a method's AUC moves a lot between
    this and `l1_logreg`, that is worth knowing before either number is quoted.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=DEFAULT_CS,
            l1_ratios=(0.0,),
            solver="saga",
            cv=inner_splits,
            scoring="roc_auc",
            max_iter=10000,
            random_state=random_state,
            use_legacy_attributes=False,
        ),
    )


def _ridge_pca(inner_splits: int, random_state: int | None) -> Pipeline:
    """PCA to at most 10 components, then plain L2 logistic regression.

    For the wide feature blocks on the single-cell axis -- a 513-column
    per-LR-pair profile over 47 patients -- where selecting among 513
    correlated columns is not a meaningful operation.
    """
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=10, random_state=random_state),
        LogisticRegression(max_iter=10000, random_state=random_state),
    )


#: Swap the classifier for every method at once by naming a different key here.
CLASSIFIERS: dict[str, ClassifierFactory] = {
    "l1_logreg": _l1_logreg,
    "l2_logreg": _l2_logreg,
    "ridge_pca": _ridge_pca,
}

DEFAULT_CLASSIFIER = "l1_logreg"


@dataclass(frozen=True)
class AUCResult:
    """One method's cross-validated phenotype AUC.

    Attributes
    ----------
    auc : float
        Mean over repeats of the pooled out-of-fold AUC. The headline number.
    auc_lo, auc_hi : float
        Percentile interval from a patient-level bootstrap of the pooled
        out-of-fold predictions. **This resamples patients holding the
        cross-validated predictions fixed**; it is the sampling variability of
        the AUC statistic, not of the whole fit. A full refit bootstrap is a
        separate, far more expensive thing (plan option O9) and is not this.
    repeat_lo, repeat_hi : float
        Range of the per-repeat pooled AUCs -- cross-validation split
        variability, which is a different quantity from `auc_lo`/`auc_hi` and is
        reported separately so the two are never conflated.
    accuracy : float
        Mean over repeats of the pooled out-of-fold accuracy at a 0.5 threshold.
    per_repeat_auc : np.ndarray
        One pooled AUC per repeat.
    per_fold_auc : np.ndarray
        Fold-level AUCs, NaN for any single-class fold. Diagnostic only.
    oof_scores : np.ndarray
        `(n_repeats, n_conditions)` out-of-fold predicted probabilities.
    y : np.ndarray
        The 0/1 coded labels, in the row order of the score matrix.
    classes : tuple[str, str]
        The label values mapped to 0 and 1, in that order.
    null_auc : float | None
        Mean AUC over label permutations, when `n_permutations > 0`. Its own
        standard error is `null_sd / sqrt(len(null_aucs))`, which is what tells
        you whether a null mean away from 0.50 is anything but noise.
    null_sd : float | None
        Standard deviation across permutation draws.
    null_p : float | None
        `(1 + #{null >= observed}) / (1 + n_permutations)`.
    null_aucs : np.ndarray | None
        The permutation distribution itself.
    n_conditions, n_components : int
        Shape of the score matrix that produced this.
    params : dict
        Everything needed to reproduce the call, for the provenance sidecar.
    """

    auc: float
    auc_lo: float
    auc_hi: float
    repeat_lo: float
    repeat_hi: float
    accuracy: float
    per_repeat_auc: np.ndarray
    per_fold_auc: np.ndarray
    oof_scores: np.ndarray
    y: np.ndarray
    classes: tuple[str, str]
    n_conditions: int
    n_components: int
    null_auc: float | None = None
    null_sd: float | None = None
    null_p: float | None = None
    null_aucs: np.ndarray | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def mean_roc(self) -> tuple[np.ndarray, np.ndarray]:
        """False- and true-positive rates of the repeat-averaged out-of-fold
        predictions, for the ROC panel."""
        fpr, tpr, _ = roc_curve(self.y, self.oof_scores.mean(axis=0))
        return fpr, tpr

    def to_row(self, **extra: Any) -> dict[str, Any]:
        """One flat record for the benchmark CSV."""
        return {
            "AUC": self.auc,
            "AUCLo": self.auc_lo,
            "AUCHi": self.auc_hi,
            "RepeatLo": self.repeat_lo,
            "RepeatHi": self.repeat_hi,
            "Accuracy": self.accuracy,
            "NullAUC": self.null_auc,
            "NullSD": self.null_sd,
            "NullN": None if self.null_aucs is None else len(self.null_aucs),
            "NullNRepeats": self.params.get("null_n_repeats"),
            "NullP": self.null_p,
            "NConditions": self.n_conditions,
            "NComponents": self.n_components,
            "Class0": self.classes[0],
            "Class1": self.classes[1],
            **extra,
        }


def encode_labels(labels: LabelLike) -> tuple[np.ndarray, tuple[str, str]]:
    """Map a two-level label to 0/1 codes, alphabetically, with the level names.

    Alphabetical ordering is what `plotGeneral.binarize_group` produces via
    pandas' categorical codes, so a sign or a class-1 identity means the same
    thing here as it does everywhere else in the package. For ALAD that puts
    `"alad"` at 0 and `"control"` at 1; the AUC is symmetric under the swap, but
    `classes` is recorded so a ROC curve is never read backwards.
    """
    s = pd.Series(list(labels), dtype=object)
    if s.isna().any():
        raise ValueError(
            f"{int(s.isna().sum())} condition(s) have no group label. Drop them "
            "from the score matrix and the labels together before scoring."
        )
    levels = sorted({str(v) for v in s})
    if len(levels) != 2:
        raise ValueError(
            f"cv_auc needs exactly two label levels, got {len(levels)}: {levels}. "
            "Collapse the column first (see `anova.prepare_group_labels`), or use "
            "the multiclass path once it exists."
        )
    y = s.map({levels[0]: 0, levels[1]: 1}).to_numpy(dtype=int)
    return y, (levels[0], levels[1])


def _usable_splits(y: np.ndarray, n_splits: int) -> int:
    """Largest fold count that keeps both classes present in every training
    fold. Stratified k-fold needs at least `n_splits` members of the rarer
    class; asking for more silently degrades."""
    smallest = int(min(np.bincount(y)))
    return max(2, min(n_splits, smallest))


def _usable_inner_splits(y_train: np.ndarray, inner_splits: int) -> int:
    """The same cap, applied to the inner CV against one training fold.

    Capping only the outer loop is not enough: the inner CV runs on a training
    fold, which holds fewer of the rarer class than the whole cohort. Asking
    `LogisticRegressionCV` for more folds than the training fold has minority
    members produces single-class inner folds, whose `roc_auc` is undefined --
    scikit-learn warns and scores them NaN, so `C` ends up selected off a
    partly-missing scoring grid. This bites on the condition sweep's small arms
    (§3.4, `N = 15` split 5/5/5) long before it bites on the full cohort.
    """
    counts = np.bincount(y_train)
    smallest = int(min(counts)) if len(counts) > 1 else 0
    return max(2, min(inner_splits, smallest))


def _bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    n_boot: int,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile interval for the AUC, resampling conditions with replacement.

    Draws that end up single-class carry no AUC and are skipped rather than
    counted as 0.5, which would drag the interval toward the null.
    """
    if n_boot <= 0:
        return (np.nan, np.nan)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, scores[idx]))
    if not aucs:
        return (np.nan, np.nan)
    tail = (100.0 - ci_level) / 2.0
    return (
        float(np.percentile(aucs, tail)),
        float(np.percentile(aucs, 100.0 - tail)),
    )


def _one_pass(
    X: np.ndarray,
    y: np.ndarray,
    factory: ClassifierFactory,
    n_splits: int,
    n_repeats: int,
    inner_splits: int,
    random_state: int,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repeated stratified CV. Returns per-repeat out-of-fold probabilities,
    per-repeat pooled AUCs and the fold-level AUCs."""
    n = len(y)
    oof = np.full((n_repeats, n), np.nan)
    fold_aucs: list[float] = []

    for r in range(n_repeats):
        seed = random_state * 1000 + r
        if groups is None:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
            splits = splitter.split(X, y)
        else:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
            splits = splitter.split(X, y, groups=groups)

        for train, test in splits:
            # Cap against *this* training fold, not the whole cohort.
            est = factory(_usable_inner_splits(y[train], inner_splits), seed)
            est.fit(X[train], y[train])
            prob = est.predict_proba(X[test])[:, 1]
            oof[r, test] = prob
            fold_aucs.append(
                roc_auc_score(y[test], prob) if len(np.unique(y[test])) == 2 else np.nan
            )

    repeat_aucs = np.array([roc_auc_score(y, oof[r]) for r in range(n_repeats)])
    return oof, repeat_aucs, np.array(fold_aucs)


def cv_auc(
    scores: np.ndarray | pd.DataFrame,
    labels: LabelLike,
    classifier: str = DEFAULT_CLASSIFIER,
    n_splits: int = 10,
    n_repeats: int = 5,
    inner_splits: int = 5,
    random_state: int = 0,
    groups: LabelLike | None = None,
    n_permutations: int = 0,
    null_n_repeats: int | None = None,
    n_boot: int = 2000,
    ci_level: float = 95.0,
) -> AUCResult:
    """Cross-validated AUC of a phenotype against one method's condition scores.

    Parameters
    ----------
    scores : np.ndarray | pd.DataFrame
        `(n_conditions, n_components)` score matrix -- CCC-RISE's read-depth
        corrected `uns["A"]`, Tensor-cell2cell's condition factor, a
        per-LR-pair profile, whatever the method produces. Rows must be in the
        same order as `labels`.
    labels : Sequence | pd.Series
        Two-level group label per condition. Collapse a multi-level column
        first; `anova.prepare_group_labels` is the shared way to do it.
    classifier : str, default="l1_logreg"
        Key into `CLASSIFIERS`. Changing this changes every method's number
        together, which is the only way a comparison stays a comparison.
    n_splits : int, default=10
        Outer folds. Reduced automatically if the rarer class is too small.
    n_repeats : int, default=5
        Outer CV repetitions, each with its own shuffle.
    inner_splits : int, default=5
        Folds of the inner CV that selects the regularization strength, fit on
        the outer training fold only.
    random_state : int, default=0
        Seed. Repeat `r` uses `random_state * 1000 + r`.
    groups : Sequence | None, optional
        Grouping for the outer splitter, one entry per condition, so that every
        row of a group falls on one side of a split. **A no-op on ALAD**, where
        each patient contributes exactly one sample -- but IBDverse does not
        have that property, so the argument exists rather than the assumption.
    n_permutations : int, default=0
        Label permutations for the null AUC. Each one repeats the whole CV, so
        this dominates runtime. The empirical p-value cannot fall below
        ``1 / (n_permutations + 1)``, so 200 is the smallest count that can
        resolve p < 0.01 and 20 the smallest that can resolve p < 0.05.
    null_n_repeats : int | None, optional
        Outer CV repeats per permutation draw. Defaults to `n_repeats`, which
        makes each draw **the same statistic** as the observed AUC it is
        compared against -- necessary for the p-value to mean anything. Setting
        it lower is a deliberate cost/validity trade: a single-repeat draw has
        larger variance than a mean of `n_repeats`, which fattens the null's
        upper tail and makes the p-value **conservative**, so a reported p is
        then an upper bound rather than an estimate. The chosen value is
        recorded in `params` and on every row.
    n_boot : int, default=2000
        Bootstrap draws for the confidence interval. Cheap -- it resamples the
        out-of-fold predictions rather than refitting.
    ci_level : float, default=95.0
        Width of the reported percentile interval.

    Returns
    -------
    AUCResult
    """
    if classifier not in CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier {classifier!r}. Available: {sorted(CLASSIFIERS)}"
        )
    factory = CLASSIFIERS[classifier]

    X = np.asarray(
        scores.to_numpy() if isinstance(scores, pd.DataFrame) else scores, dtype=float
    )
    if X.ndim != 2:
        raise ValueError(
            f"scores must be 2D (n_conditions, n_components), got {X.shape}"
        )
    y, classes = encode_labels(labels)
    if len(y) != X.shape[0]:
        raise ValueError(
            f"scores has {X.shape[0]} rows but {len(y)} labels were given; they must "
            "be aligned condition-for-condition."
        )
    if not np.isfinite(X).all():
        raise ValueError(
            "scores contains non-finite values. A method that cannot score a "
            "condition should be dropped from the comparison explicitly, not "
            "imputed here."
        )

    groups_arr = None if groups is None else np.asarray(list(groups))
    splits = _usable_splits(y, n_splits)

    oof, repeat_aucs, fold_aucs = _one_pass(
        X, y, factory, splits, n_repeats, inner_splits, random_state, groups_arr
    )

    pooled = oof.mean(axis=0)
    rng = np.random.default_rng(random_state)
    lo, hi = _bootstrap_ci(y, pooled, n_boot, ci_level, rng)

    accuracy = float(
        np.mean(
            [accuracy_score(y, (oof[r] > 0.5).astype(int)) for r in range(n_repeats)]
        )
    )

    null_repeats = n_repeats if null_n_repeats is None else int(null_n_repeats)
    null_auc = null_p = null_sd = None
    null_aucs = None
    if n_permutations > 0:
        perm_rng = np.random.default_rng(random_state + 1)
        vals = []
        for _ in range(n_permutations):
            y_perm = perm_rng.permutation(y)
            _, perm_repeat, _ = _one_pass(
                X,
                y_perm,
                factory,
                splits,
                null_repeats,
                inner_splits,
                random_state,
                groups_arr,
            )
            # Mean over the draw's repeats, matching how the observed AUC is
            # formed, so observed and null are the same statistic.
            vals.append(float(perm_repeat.mean()))
        null_aucs = np.array(vals)
        null_auc = float(null_aucs.mean())
        null_sd = float(null_aucs.std(ddof=1)) if len(null_aucs) > 1 else float("nan")
        observed = float(repeat_aucs.mean())
        null_p = float((1 + np.sum(null_aucs >= observed)) / (1 + n_permutations))

    return AUCResult(
        auc=float(repeat_aucs.mean()),
        auc_lo=lo,
        auc_hi=hi,
        repeat_lo=float(repeat_aucs.min()),
        repeat_hi=float(repeat_aucs.max()),
        accuracy=accuracy,
        per_repeat_auc=repeat_aucs,
        per_fold_auc=fold_aucs,
        oof_scores=oof,
        y=y,
        classes=classes,
        n_conditions=int(X.shape[0]),
        n_components=int(X.shape[1]),
        null_auc=null_auc,
        null_sd=null_sd,
        null_p=null_p,
        null_aucs=null_aucs,
        params={
            "classifier": classifier,
            "n_splits": splits,
            "n_splits_requested": n_splits,
            "n_repeats": n_repeats,
            "inner_splits": inner_splits,
            "random_state": random_state,
            "n_permutations": n_permutations,
            "null_n_repeats": null_repeats,
            "null_matched": null_repeats == n_repeats,
            "n_boot": n_boot,
            "ci_level": ci_level,
            "grouped": groups_arr is not None,
        },
    )
