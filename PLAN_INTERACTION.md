# PLAN_INTERACTION.md — Selecting the CPD rank for the interaction tensor

**Audience:** an agent tasked with exploring, implementing, and evaluating rank-selection
criteria for the CP decomposition of the 4D cell–cell communication tensor.

**Deliverable:** a written report recommending a criterion for the CPD rank, backed by results
computed on this repository's data. Do not modify the manuscript.

**Scope:** this plan covers *only* the CPD stage — the decomposition of the
(condition × sender eigen-state × receiver eigen-state × LR-pair) tensor. The RISE rank is
handled separately in `PLAN_RISE.md`; take the RISE rank as given here (but see §4 on the joint
sweep). Where the two plans overlap methodologically (permutation nulls, replicable-component
counts, degeneracy diagnostics), reuse the implementation rather than duplicating it.

---

## 1. Context

Reviewer #2, major comment 4: *"it remains unclear why 35 RISE components and 8 CPD components
were chosen rather than, for example, 30–50 RISE components or 6–12 CPD components. The metrics
do not provide clear guidance… The authors should provide consistent and sharp metrics."*
Reviewer #1, major comment 4, separately objects that for ALAD the ranks were chosen against
predictive performance on the full dataset, which optimistically biases the reported AUC.

Current evidence for the CPD rank lives in `figures/figureS1c_d.py` (COVID) and
`figures/figureS5f_g.py` (ALAD), computed by `run_fms_r2x_analysis` in
`cellcommunicationpf2/tensor.py`: in-sample R2X (monotone, no elbow) and a bootstrap FMS where
the bootstrap resamples the **last tensor dimension** (`resample_tensor`, i.e. LR pairs) and
matches with `skip_mode=3`. Neither gives a cut point.

The CPD stage is more tractable than the RISE stage: the tensor is small, dense, rectangular,
and fits in memory, so cross-validation and permutation schemes that are expensive at the RISE
stage are cheap here. **Expect to be able to give a much sharper answer for this rank.**

---

## 2. Existing code to reuse

| Thing | Where |
|---|---|
| Build the 4D tensor from a RISE-processed AnnData | `calculate_interaction_tensor` in `cellcommunicationpf2/tensor.py` |
| CP fit | `tensorly.decomposition.parafac` (as used in `run_fms_r2x_analysis`) |
| R2X for a CP model | `calculate_r2x` in `cellcommunicationpf2/ccc.py` |
| Current FMS between two CP models | `calculate_fms_cpd` in `cellcommunicationpf2/tensor.py` |
| Bootstrap over the LR mode | `resample_tensor` in `cellcommunicationpf2/tensor.py` |
| Existing sweep driver | `run_fms_r2x_analysis` in `cellcommunicationpf2/tensor.py` |
| Data-fraction sensitivity sweep | `run_fms_r2x_data_percentage_analysis` in `cellcommunicationpf2/tensor.py` |
| Disk caching | `load_or_compute_dataframe` in `cellcommunicationpf2/factors.py` |
| Cached AnnData with factors | `get_covid_data(project_cells=True)`, `get_alad_data(project_cells=True)` in `figures/common.py` |

Note that the tensor is built from `adata.obsm["projections"]`, so it depends on the RISE fit;
regenerating it requires a RISE model to be loaded or refit.

Run everything with `uv run`. `make test`, `make ty`. Keep ruff clean.

---

## 3. Approaches to evaluate

### Approach A — Missing-entry cross-validation (primary recommendation)

**Goal:** a metric with a true interior minimum, unlike in-sample R2X.

Mask a random 5–10% of the entries of the 4D tensor, fit CP with the mask, and score
reconstruction on the **held-out entries only**. Averaged over several masks, this rises once
the extra components start fitting noise, giving a genuine optimum.

Unlike the RISE stage, this is easy here: `tensorly` supports masked CP fitting
(`parafac(..., mask=...)`, and/or the EM-style imputation loop where masked entries are filled
from the current model each ALS sweep). Verify which variant the installed tensorly version
implements and confirm on synthetic data that the held-out error is U-shaped around a known
true rank before trusting it on real data.

Details to get right:
- Mask **entries**, not whole slabs, unless you specifically want a structured holdout.
- Use several independent masks (≥5) and report mean ± spread; a single mask is noisy.
- Score on held-out entries with the same normalization as `calculate_r2x` so the numbers are
  comparable to the existing R2X figure.
- Consider a structured variant as a secondary check: hold out entire **conditions** (samples)
  and score how well the shared sender/receiver/LR factors reconstruct an unseen condition after
  fitting only its condition loading. This is the version that speaks most directly to
  generalization across patients.

This is the single strongest answer to "provide consistent and sharp metrics," and it is
unsupervised — it does not touch ALAD status, which is what makes it usable to replace the
current predictive-performance-based rank choice (see §5).

### Approach B — Permutation null / parallel analysis

**Goal:** a crossing point instead of a smooth curve, plus a null model for significance.

Refit CP at each rank on tensors whose communication structure has been destroyed but whose
marginals are preserved. Candidate nulls, which test different things — implement at least two
and report the difference:
- Permute the LR-pair mode independently within each (condition, sender, receiver) fiber.
- Permute sender/receiver eigen-state labels independently per condition.
- Shuffle cell-to-eigen-state assignment upstream and rebuild the tensor (a stronger null that
  also destroys the eigen-state structure the tensor is built on).

Then select the largest rank whose marginal ΔR2X exceeds the null's marginal ΔR2X.

**This does double duty.** Reviewer #1's major comment 6 asks for a null model / significance
assessment for whether a component's communication exceeds chance — CCC-RISE currently has
none, and CellPhoneDB/CellChat do. The same permutation machinery yields per-component empirical
p-values (compare each observed component's explained variance, or its top LR-pair loadings,
against the null distribution). Build it with both uses in mind and say so in the report; this
is likely the highest-leverage single piece of work in this plan.

### Approach C — Core consistency diagnostic (CORCONDIA)

Classic CP rank diagnostic: fit at rank `r`, compute the Tucker3 core implied by the CP factors,
and measure how close it is to the superdiagonal. It stays near 100 while the model is genuinely
multilinear and falls off a cliff when over-factored. Cheap to compute for a 4-way tensor of
this size.

State the known caveat up front in the report: CORCONDIA is pessimistic under high noise and
often recommends a rank lower than what is useful. Use it as corroboration, not as the sole
criterion. If it happens to give a clean cliff at or near 8, it is a very quotable result.

### Approach D — Count of replicable components, and better resampling

Same argument as in `PLAN_RISE.md` §3, approach 4: mean FMS must decay smoothly because it
averages strong components against marginal ones. Report instead the number of components whose
matched similarity exceeds τ ∈ {0.6, 0.7, 0.8}.

Additionally, reconsider **what** is being resampled. `resample_tensor` bootstraps the last
dimension (LR pairs), which tests robustness to the LR database. That is a reasonable question
but not the most important one. Also evaluate:
- Bootstrap over **conditions** (patients) — tests whether programs replicate across people.
  This is the claim the biology rests on.
- Split-half over conditions.
- Bootstrap of **cells** upstream, rebuilding the tensor each time — the most complete but most
  expensive, since it requires refitting or re-projecting RISE.

Report which resampling axis produces the most discriminating curve, and be explicit that
different axes answer different questions.

### Approach E — Degeneracy and multi-start diagnostics

Over-factored CP models exhibit two-factor degeneracies. At each rank, from several random
initializations (`svd_init="random"`, varying seeds):
- Maximum pairwise **triple/quadruple cosine congruence** between components — the onset of a
  congruence approaching −1 is the classic degeneracy signature and a hard upper bound on
  usable rank.
- Whether independent initializations converge to the same solution and the same final R2X.
- Condition number of each factor matrix; spread of component weights.

This is cheap, needs no holdout, and simultaneously satisfies Reviewer #2's request for
random-seed sensitivity. Note that the existing figures use `svd_init="random"` for COVID
(`figureS1c_d.py`) — check whether ALAD does the same, since an init mismatch between datasets
would itself explain some of the inconsistency the reviewer noticed.

---

## 4. Joint (RISE × CPD) rank selection

Reviewer #2 asks why *both* 35 and 8. The two ranks interact: the CPD tensor's sender/receiver
dimensions are literally the RISE rank, so the sensible CPD rank depends on the RISE rank.

Produce a **2-D heatmap of held-out imputation error (approach A) over a grid of
(RISE rank × CPD rank)**, for both datasets. Even a coarse grid — RISE ∈ {20, 30, 35, 40, 50},
CPD ∈ {4, 6, 8, 10, 12, 16} — is far more convincing than two separate 1-D curves, and it lets
you show whether the choice sits in a broad flat basin (the likely and perfectly defensible
outcome) or on a knife edge. Coordinate with whoever is executing `PLAN_RISE.md` so the RISE
axis uses the same fits.

Budget this carefully: it is the most expensive item here. Do it last, at coarse resolution,
and only after approach A is validated.

---

## 5. Consequence for the ALAD predictive analysis

Reviewer #1 major comment 4 is that the ALAD ranks were chosen for "optimal predictive
performance for ALAD status" outside the cross-validation folds, which biases the reported AUC.
There are two ways out and the report should recommend one:

- **(Preferred) Select the ranks by an unsupervised criterion** — approach A — and *then* report
  AUC. The rank is no longer chosen against the outcome, so the bias objection evaporates and no
  nested CV is required. If the unsupervised criterion picks ranks near the published ones, this
  is a clean and cheap resolution; quantify how much AUC changes at the unsupervised-selected
  ranks.
- **(Fallback) Nested cross-validation**: move rank selection inside the outer CV folds and
  re-report AUC. Correct but expensive, and it will lower the headline number.

Estimate the AUC at each rank on the grid as part of the sweep, so the report can state exactly
what the honest number is under either scheme. Do not let this predictive evaluation contaminate
the criterion itself — compute the unsupervised criterion first and record the selected rank
before looking at any AUC.

---

## 6. Datasets, compute, deliverables

Run on both COVID-19 BAL and ALAD, with identical procedure and thresholds — the fact that the
two datasets currently use different metrics is itself one of the reviewer's complaints. Sweep
CPD ranks over at least {1…16} to bracket the published choice of 8 and the reviewer's suggested
6–12 window.

Cache all sweeps to `cellcommunicationpf2/data/` via `load_or_compute_dataframe`, following the
existing file-naming pattern. Keep scratch work out of the repo.

Deliverables, written to the repository root:

1. **`INTERACTION_RANK_REPORT.md`**:
   - Recommendation up front: which criterion, which rank per dataset, confidence.
   - One section per approach with an explicit verdict (useful / not useful / inconclusive).
   - Table of recommended rank per criterion per dataset.
   - The (RISE × CPD) heatmap and what it shows about how flat the optimum is.
   - Explicit statement of whether the published rank of 8 survives, and if not, what would need
     refitting. Report this plainly — do not tune criteria toward the published answer.
   - A note on how the approach-B null doubles as the per-component significance test requested
     by Reviewer #1 comment 6, with a sketch of what that analysis would look like.
   - Recommendation for the accompanying sensitivity analysis: match components across ranks
     6/8/12 and show that top LR pairs, condition-factor patterns, and the biological
     conclusions are preserved. This argument matters as much as the criterion itself, because
     it reframes the issue from "you picked the wrong number" to "the conclusions do not depend
     on the number."
   - A draft paragraph, ~150 words, for the response to Reviewer #2 major comment 4 (CPD portion)
     and a short note for Reviewer #1 comment 4.
2. **Reusable code** in `cellcommunicationpf2/` (extend the `rank_selection.py` module from
   `PLAN_RISE.md` if it exists), with pytest coverage — in particular a synthetic-tensor test
   that held-out imputation error is minimized at the true rank.
3. **Cached sweep CSVs** and **draft figure modules** in the existing `makeFigure()` idiom, under
   provisional names that do not collide with existing figures.

Do **not** edit `manuscript/review-response.qmd` or any manuscript source. Do not commit.

---

## 7. Pitfalls

- **Sign/permutation ambiguity**: CP components come out in arbitrary order and sign. Always
  match with `linear_sum_assignment` on `|cos|`; never compare positionally.
- **Convergence vs. instability**: before interpreting a low FMS or a jumpy curve as
  instability, check that the fits converged (iteration count, final error, `tol`). The RISE
  cached sweeps show occasional catastrophic outliers that look like convergence failures; the
  same may be true here.
- **The tensor is non-negative only if the communication scores are.** RISE factors are not
  sign-constrained, so the tensor contains negative entries (this is Reviewer #1 comment 3 and
  Reviewer #2 minor comment 3). Do not silently assume non-negativity in any null model or
  masking scheme; and if you evaluate non-negative CP as an alternative, treat it as a separate
  question and flag it, since it changes the model, not just the rank.
- **Normalization**: be consistent about how the tensor is scaled before decomposition, and make
  sure held-out error and in-sample R2X use the same convention, or the curves will not be
  comparable.
