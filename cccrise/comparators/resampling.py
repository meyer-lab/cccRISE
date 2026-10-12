"""Resampling helpers behind the ALAD comparator bootstrap
(`validation.comparator_bootstrap`, `figureS_comparator_clinical`).

They are kept here as that analysis ran them: `main` replaced the bootstrap
stability pipeline they came from (`utils.resample_conditions`,
`validation.anova_stability`) with a jackknife (#223) whose
`align_to_reference_components` sign-corrects values, which this bootstrap
deliberately does not."""

import warnings
from collections.abc import Sequence
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

Factors = list[np.ndarray]

#: Modes `align_to_reference_components` matches a replicate to the reference
#: over. Only the ligand-receptor mode (3) survives a separate RISE fit with
#: its coordinates intact: the condition mode indexes a different patient
#: roster, and the sender/receiver modes are expressed in that replicate's own
#: RISE eigen-states.
BOOTSTRAP_MATCH_MODES: tuple[int, ...] = (3,)


#: Column order of one replicate's table, and so of the replicate-level table
#: once `bootstrap_anova` prepends 'Replicate'.
REPLICATE_COLUMNS = [
    "Component",
    "MatchedComponent",
    "EtaSquared",
    "F",
    "P",
    "GroupMeanDiff",
    "MatchScore",
    "SignedCosine",
    "RunnerUpScore",
    "R2XCP",
]


def _paired_conditions(
    obs: pd.DataFrame, condition_name: str, patient_by: str, labels: np.ndarray
) -> dict[str, str]:
    """Map each condition sharing a patient with exactly one sibling condition
    to that sibling's label, for `resample_conditions`'s `patient_by`.

    A patient contributing only one condition has no sibling and so is absent
    from the returned mapping. A patient contributing more than two is paired
    cyclically (each condition maps to the "next" one), which is untested
    territory -- ALAD's own paired patients each contribute exactly two.
    """
    patients = obs[patient_by].to_numpy().astype(str)
    table = pd.DataFrame(
        {condition_name: labels, patient_by: patients}
    ).drop_duplicates(condition_name)

    pairs: dict[str, str] = {}
    for _patient, members in table.groupby(patient_by)[condition_name]:
        conds = members.tolist()
        if len(conds) < 2:
            continue
        for i, cond in enumerate(conds):
            pairs[cond] = conds[(i + 1) % len(conds)]
    return pairs


def _carry_paired_samples(
    drawn: np.ndarray | Sequence[str],
    pair_of: dict[str, str],
    rng: np.random.Generator,
    stratum_of: dict[str, Any] | None = None,
) -> list[str]:
    """Carry in a paired condition's sibling sample, dropping one other random
    draw per carry so the resample's total size is unchanged.

    Some ALAD patients contributed two samples (conditions), so an ordinary
    per-condition bootstrap can draw one of a patient's samples without the
    other, treating one enrollment as two independent ones. This carries the
    sibling sample in whenever the draw hits either one; the compensating
    drop is what keeps the final number of patients in the resampling fixed,
    rather than growing by one every time a pair is hit.

    The drop never takes the draw that triggered a carry or a carried-in
    sibling (so a carry is not undone, by its own drop or a later one),
    unless nothing else is left to drop. A draw that an earlier carry
    dropped triggers no carry of its own.

    With `stratum_of` (condition -> stratum), the drop is drawn from the
    carried sibling's own stratum, so every stratum keeps its original count
    even when a patient's two samples fall in different strata.

    Each of the original draws is visited once, so a condition drawn twice
    triggers two independent carry-and-drop steps.
    """
    # [condition, protected]; lists, so entries are told apart by identity
    entries = [[cond, False] for cond in drawn]
    for entry in list(entries):
        sibling = pair_of.get(entry[0])
        if sibling is None or not any(e is entry for e in entries):
            continue
        entry[1] = True
        entries.append([sibling, True])
        _drop_one_unprotected(entries, sibling, rng, stratum_of)
    return [cond for cond, _ in entries]


def _drop_one_unprotected(
    entries: list[list[Any]],
    sibling: str,
    rng: np.random.Generator,
    stratum_of: dict[str, Any] | None,
) -> None:
    """Drop one random entry from `sibling`'s stratum, preferring entries a
    carry has not protected (see `_carry_paired_samples`)."""
    in_stratum = [
        i
        for i, (c, _) in enumerate(entries)
        if stratum_of is None or stratum_of[c] == stratum_of[sibling]
    ]
    pool = [i for i in in_stratum if not entries[i][1]] or in_stratum
    entries.pop(pool[rng.integers(0, len(pool))])


def resample_conditions(
    data: anndata.AnnData,
    condition_name: str,
    stratify_by: str | None = None,
    random_seed: int | None = None,
    patient_by: str | None = None,
) -> anndata.AnnData:
    """Bootstrap whole conditions (patients) with replacement, keeping every
    cell of a drawn condition.

    Complements `resample_cells`, which resamples cells *within* each condition
    and so holds the patient roster fixed. Here the roster itself is the
    resampled unit, which is what a bootstrap of a condition-mode result -- a
    per-patient factor, or a model fit on one -- requires: the sampling
    variability being estimated is variability in which patients were enrolled.

    A condition drawn more than once is relabeled (``<label>__rep2``, ...) so
    stages keying on `condition_name` -- `import_data.add_cond_idxs`, and
    through it RISE's per-condition slicing -- treat repeat draws as separate
    conditions instead of merging their cells into one oversized slice.

    Parameters
    ----------
    data : anndata.AnnData
        Single-cell expression data to resample.
    condition_name : str
        Column in `data.obs` identifying the condition (patient) to draw.
    stratify_by : str | None, optional
        Column in `data.obs` that is constant within a condition, typically a
        group or outcome label. When given, conditions are drawn with
        replacement *within* each stratum, so every stratum keeps exactly its
        original condition count. Without it, a uniform draw on a small cohort
        can return a replicate whose groups are badly imbalanced, or that has
        lost a group entirely -- which changes what the condition factor can
        express, and would register as instability caused by the draw rather
        than by the cohort.
    random_seed : int | None, optional
        Seed for the draw. None leaves it unseeded.
    patient_by : str | None, optional
        Column in `data.obs` that is constant within a condition and shared by
        every condition (sample) belonging to the same patient. When given,
        drawing one of a two-sample patient's conditions carries its sibling
        condition in too, and drops one other random draw to compensate --
        never the pair itself unless nothing else is left, and from the
        sibling's own stratum when `stratify_by` is given (see
        `_carry_paired_samples`). Conditions whose patient contributes only
        one of them are unaffected.

    Returns
    -------
    anndata.AnnData
        A copy holding the drawn conditions' cells, with `condition_name`
        relabeled per draw and `var["means"]` recomputed over the resampled
        cells -- see `resample_cells` for why that column has to be refreshed.
    """
    rng = np.random.default_rng(random_seed)
    obs = cast(pd.DataFrame, data.obs)
    labels = obs[condition_name].to_numpy().astype(str)

    stratum_of = None
    if stratify_by is None:
        conditions = np.unique(labels)
        drawn = rng.choice(conditions, size=conditions.size, replace=True)
    else:
        strata = (
            pd.DataFrame({condition_name: labels, stratify_by: obs[stratify_by].values})
            .drop_duplicates(condition_name)
            .set_index(condition_name)[stratify_by]
        )
        drawn = np.concatenate(
            [
                rng.choice(members.index.to_numpy(), size=len(members), replace=True)
                for _, members in strata.groupby(strata, observed=True)
            ]
        )
        stratum_of = strata.to_dict()

    if patient_by is not None:
        pair_of = _paired_conditions(obs, condition_name, patient_by, labels)
        drawn = _carry_paired_samples(drawn, pair_of, rng, stratum_of)

    cells_of = {cond: np.flatnonzero(labels == cond) for cond in np.unique(labels)}
    draws_so_far: dict[str, int] = {}
    cell_indices, new_labels = [], []
    for cond in drawn:
        draws_so_far[cond] = draws_so_far.get(cond, 0) + 1
        cells = cells_of[cond]
        suffix = "" if draws_so_far[cond] == 1 else f"__rep{draws_so_far[cond]}"
        cell_indices.append(cells)
        new_labels.append(np.full(cells.size, f"{cond}{suffix}"))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Observation names are not unique")
        resampled = data[np.concatenate(cell_indices)].copy()
    resampled.obs[condition_name] = np.concatenate(new_labels)
    resampled.obs_names_make_unique()
    resampled.var["means"] = np.asarray(cast(Any, resampled.X).mean(axis=0)).ravel()
    return resampled


def _signed_mode_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed cosine similarity between every column of `a` and every column of `b`."""
    a_norm = a / np.linalg.norm(a, axis=0, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=0, keepdims=True)
    return a_norm.T @ b_norm


def _runner_up_score(combined: np.ndarray, i: int, j: int) -> float:
    """Row `i`'s best score other than its assigned column `j` (NaN when
    there is no other column)."""
    if combined.shape[1] <= 1:
        return np.nan
    alt_scores = combined[i].copy()
    alt_scores[j] = -np.inf
    return float(alt_scores.max())


def align_to_reference_components(
    reference_factors: Factors,
    factors: Factors,
    values: np.ndarray,
    modes: tuple[int, ...] = BOOTSTRAP_MATCH_MODES,
) -> pd.DataFrame:
    """Relabel one CP fit's per-component values by the reference fit's
    component numbering.

    CP does not fix component order across separate fits, so a per-component
    quantity is not comparable between two fits until each of the second
    fit's components has been identified with a reference component. Optimal
    assignment on the |cosine| of the `modes` gives the relabeling (several
    modes are scored by their mean |cosine|). Values are not sign-corrected:
    CP can flip any two modes together, so the match mode's sign does not
    determine the sign of the mode a value comes from. Values from the
    condition factor are already oriented consistently by
    `ccc_rise.standardize_cc_pf2` (positive column mean).

    Parameters
    ----------
    reference_factors : list[np.ndarray]
        The reference fit's CP factors [A, B, C, D]; its component numbering
        is the one every aligned fit is expressed in.
    factors : list[np.ndarray]
        The fit being aligned, sharing `reference_factors`' dimensions along
        every mode in `modes`.
    values : np.ndarray
        One value per component of `factors`, in that fit's own order.
    modes : tuple[int, ...], default=BOOTSTRAP_MATCH_MODES
        Modes the match is computed over; defaults to the ligand-receptor
        mode alone.

    Returns
    -------
    pd.DataFrame
        One row per reference component matched (1-indexed), columns
        'Component', 'MatchedComponent', 'Value', 'MatchScore',
        'SignedCosine' (a match diagnostic, NaN when matching on several
        modes), 'RunnerUpScore'.
    """
    signed_cos_by_mode = {
        mode: _signed_mode_cosine(reference_factors[mode], factors[mode])
        for mode in modes
    }
    combined = np.mean([np.abs(cos) for cos in signed_cos_by_mode.values()], axis=0)

    row_ind, col_ind = linear_sum_assignment(-combined)

    rows = []
    for i, j in zip(row_ind, col_ind, strict=True):
        runner_up_score = _runner_up_score(combined, i, j)
        signed_col = modes[0] if len(modes) == 1 else None
        signed_cos = (
            signed_cos_by_mode[signed_col][i, j] if signed_col is not None else np.nan
        )

        rows.append(
            {
                "Component": i + 1,
                "MatchedComponent": j + 1,
                "Value": float(values[j]),
                "MatchScore": float(combined[i, j]),
                "SignedCosine": float(signed_cos),
                "RunnerUpScore": runner_up_score,
            }
        )

    return pd.DataFrame(rows).sort_values("Component").reset_index(drop=True)


def summarize_bootstrap_anova(
    boot_anova: pd.DataFrame,
    reference_anova: pd.DataFrame,
    ci_level: float = 95.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """One row per component: its bootstrap effect-size distribution, and how
    reliably the association and its direction survive resampling.

    Three things are reported side by side rather than collapsed into one
    number, because they call for different conclusions: how large the effect
    is and how tightly the rosters pin it down (the eta-squared interval); how
    often the association reaches `alpha` at all (`FracSignificant`); and
    whether the component was even re-identifiable across replicates
    (`MedianMatchScore`/`MinMatchScore`), which is a CP identifiability
    question rather than a question about the phenotype.

    There is deliberately no "interval excludes zero" column: eta-squared is a
    variance ratio and cannot be negative, so such a test would be vacuous.
    `FracSignificant` takes its place -- read as a descriptive tally, since
    repeated patients in a resample make per-replicate p-values
    anti-conservative.

    Parameters
    ----------
    boot_anova : pd.DataFrame
        Output of `bootstrap_anova`.
    reference_anova : pd.DataFrame
        The full-cohort fit's per-component ANOVA table -- `fit_anova_pipeline`'s
        ``result["anova"]`` -- carrying 'Component', 'EtaSquared', 'F', 'P' and
        'GroupMeanDiff'.
    ci_level : float, default=95.0
        Central percentile interval width, in percent, taken on eta-squared.
    alpha : float, default=0.05
        Threshold `FracSignificant` counts replicates against.

    Returns
    -------
    pd.DataFrame
        One row per component, sorted by component, with columns 'Component',
        'ReferenceEtaSquared', 'ReferenceF', 'ReferenceP', 'MeanEtaSquared',
        'MedianEtaSquared', 'SDEtaSquared', 'CILow', 'CIHigh', 'MedianP',
        'FracSignificant', 'ReferenceGroupMeanDiff', 'MedianGroupMeanDiff',
        'FracSameSignAsReference' (NaN when the reference group-mean
        difference is exactly zero, i.e. has no direction to agree with),
        'MedianMatchScore', 'MinMatchScore', 'NReplicates'.
    """
    tail = (100.0 - ci_level) / 2.0
    reference_by_component = reference_anova.set_index("Component")
    rows = []
    for component_key, group in boot_anova.groupby("Component"):
        component = cast(int, component_key)
        eta = group["EtaSquared"].to_numpy(dtype=float)
        p_values = group["P"].to_numpy(dtype=float)
        diffs = group["GroupMeanDiff"].to_numpy(dtype=float)
        reference = reference_by_component.loc[component]
        reference_diff = float(reference["GroupMeanDiff"])
        ci_low, ci_high = np.nanpercentile(eta, [tail, 100.0 - tail])
        rows.append(
            {
                "Component": component,
                "ReferenceEtaSquared": float(reference["EtaSquared"]),
                "ReferenceF": float(reference["F"]),
                "ReferenceP": float(reference["P"]),
                "MeanEtaSquared": float(np.nanmean(eta)),
                "MedianEtaSquared": float(np.nanmedian(eta)),
                "SDEtaSquared": float(np.nanstd(eta, ddof=1)) if len(eta) > 1 else 0.0,
                "CILow": float(ci_low),
                "CIHigh": float(ci_high),
                "MedianP": float(np.nanmedian(p_values)),
                # A replicate whose ANOVA came back degenerate (NaN p) counts
                # as not significant rather than dropping out of the tally.
                "FracSignificant": float(np.mean(p_values < alpha)),
                "ReferenceGroupMeanDiff": reference_diff,
                "MedianGroupMeanDiff": float(np.nanmedian(diffs)),
                "FracSameSignAsReference": (
                    float(np.mean(np.sign(diffs) == np.sign(reference_diff)))
                    if np.isfinite(reference_diff) and reference_diff != 0
                    else np.nan
                ),
                "MedianMatchScore": float(np.median(group["MatchScore"])),
                "MinMatchScore": float(np.min(group["MatchScore"])),
                "NReplicates": len(eta),
            }
        )
    return pd.DataFrame(rows).sort_values("Component").reset_index(drop=True)
