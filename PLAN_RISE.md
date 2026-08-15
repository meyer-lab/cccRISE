# PLAN_RISE.md — Selecting the RISE / PARAFAC2 rank

**Audience:** an agent tasked with exploring, implementing, and evaluating rank-selection
criteria for the RISE (PARAFAC2) stage of CCC-RISE. Read this whole file before writing code.

**Deliverable:** a written report recommending *one primary criterion* (plus at most two
supporting diagnostics) for choosing the RISE rank, backed by results actually computed on
this repository's data. Do not modify the manuscript.

---

## 1. Context and motivation

CCC-RISE has two ranks: the RISE/PARAFAC2 rank (number of eigen-states) and the CPD rank
(number of communication programs). **This plan concerns only the RISE rank, and every
criterion considered here must be computable from the single-cell expression data alone** —
no ligand–receptor tensor, no CPD, no downstream classifier, no clinical outcome. The reason
is twofold: (a) selecting the RISE rank against a downstream predictive objective is exactly
the optimistic-bias problem Reviewer #1 raised (major comment 4), and (b) a criterion that
depends on the CCC tensor cannot be presented as an independent justification of the basis.

Reviewer #2 (major comment 4) objects that the current metrics "do not provide clear guidance"
and that different metrics were used for the two datasets. The current state of the evidence:

- **R2X** (`cellcommunicationpf2/data/figS1a_b_rise_stability.csv`) climbs monotonically and
  smoothly — 0.026 at rank 1 to 0.056 at rank 19 on COVID. No elbow, and the absolute values
  are low enough that a reader cannot tell how much of it is structure at all.
- **Bootstrap FMS** (same file, `calculate_fms_rise` in `cellcommunicationpf2/tensor.py`)
  decays smoothly with rank: ~0.996 through rank 5, ~0.86 by rank 9, with high run-to-run
  variance (e.g. rank 4 run 0 gives 0.75 while runs 1–2 give >0.995).

Neither yields a defensible cut point. This plan explores five alternatives.

**Important framing to carry into the report:** the RISE rank is a *basis size*, closer in
character to "how many principal components" than to "how many clusters." It is legitimate for
the honest answer to be "there is a broad plateau, here is a principled criterion that lands
inside it, and here is a sensitivity analysis showing the conclusions do not depend on the
exact choice." Do not manufacture a sharp optimum that the data does not support — a negative
result about a given metric is a useful finding, and should be reported as such.

---

## 2. Existing code you should reuse

Do not rewrite what exists. Relevant entry points:

| Thing | Where |
|---|---|
| PARAFAC2 fit, returns `((weights, [A, B, C], projections), R2X)` | `parafac2.parafac2.parafac2_nd` |
| Thin wrapper storing results into AnnData | `rise_store_r2x` in `cellcommunicationpf2/tensor.py` |
| Projection/error step against **fixed** factors | `parafac2.utils.project_data(X_list, factors, norm_X_sq, mode, return_projections=False)` |
| AnnData → per-condition `SampleArray` list | `parafac2.utils.anndata_to_list` |
| Sign/permutation standardization | `parafac2.utils.standardize_pf2` |
| Current FMS between two fitted RISE models | `calculate_fms_rise` in `cellcommunicationpf2/tensor.py` |
| Stratified bootstrap of cells within sample | `resample` in `cellcommunicationpf2/utils.py` |
| Disk-cached dataframe computation | `load_or_compute_dataframe` in `cellcommunicationpf2/factors.py` |
| Data loaders | `import_balf_covid`, `import_alad`, `add_cond_idxs` in `cellcommunicationpf2/import_data.py` |
| Existing rank-sweep figures (patterns to imitate) | `figures/figureS1a_b.py`, `figures/figureS5f_g.py` |

Key facts about the model that constrain what is and is not a valid holdout:

- The model is `X_i ≈ P_i diag(a_i) B Cᵀ`, where `X_i` is (cells_i × genes) for condition `i`,
  `P_i` is (cells_i × rank) with orthonormal columns, `B` is (rank × rank), `C` is
  (genes × rank), and `a_i` is row `i` of `A`.
- `B`, `C` are **shared** across conditions; `P_i` and `a_i` are **per-condition**.
- Critically, `P_i` has one row per cell. **A held-out cell has no projection until one is fit
  from its own expression.** This is the single most important design constraint below and is
  where a naive cross-validation scheme leaks.
- `project_data` computes both the optimal projections and the accumulated squared error given
  fixed factors, so "score this data under a frozen model" is already a solved subproblem.
- `X` is mean-centered implicitly via `X_in.var["means"]` inside `anndata_to_list`; account for
  this when you compute your own reconstruction errors, or you will get nonsense.

Run everything with `uv run`. `make test` runs pytest; `make ty` type-checks. Keep formatting
clean with ruff.

---

## 3. Approaches to evaluate

Investigate all five. They are ordered by expected payoff, not by required order of work — if
approach 1 proves expensive, get 3 and 4 done first since they are cheap.

### Approach 1 — Cross-validated reconstruction error

**Goal:** a metric that is genuinely U-shaped in rank, unlike in-sample R2X.

There are two variants, and **the difference between them is the crux of this approach**.

**1a. Leave-one-sample-out (cheap, but likely flawed).** Fit RISE on all samples but one; then
for the held-out sample, freeze `B` and `C` and use `project_data` to obtain its projections
and reconstruction error. Score R2X on the held-out sample only. Implementation is
straightforward: build a one-element `X_list` for the held-out sample and pass factors with `A`
restricted to that condition's row (which must itself be estimated — see the caveat).

*Caveat you must confront:* because `P_i` is refit per cell from the held-out sample's own
data, and `a_i` likewise, the held-out sample is allowed to choose its own optimal projection
into the shared subspace. Increasing the rank strictly enlarges that subspace, so this quantity
may well be **monotonically increasing in rank just like in-sample R2X**, in which case it
answers nothing. Test this empirically first — it is a ~1 hour experiment and it determines
whether the rest of the approach is worth pursuing. Report the answer either way.

**1b. Gene × cell block holdout (bi-cross-validation; rigorous).** This is the version that
cannot leak:

1. Partition genes into a training set `G_tr` and a held-out set `G_te` (say 90/10).
2. Partition cells within each condition into halves `N_a` and `N_b`.
3. Fit RISE at rank `r` using only `G_tr` (all cells). This yields `A`, `B`, `C_tr`, and a
   projection row `P_i[n]` for **every** cell `n` — note these projections are derived from
   training genes only.
4. Estimate held-out-gene loadings `C_te` by least squares using only cells in `N_a`:
   solve `X[N_a, G_te] ≈ (P[N_a] diag(a) B) C_teᵀ` for `C_te`.
5. Predict `X[N_b, G_te]` using `P[N_b]` (from step 3, training genes) and `C_te` (from step 4,
   `N_a` cells). Score error on this block.

No parameter in the predicted block was fit using that block, so the error must eventually rise
with rank. This is the metric most likely to give a real optimum and the one most defensible to
a reviewer. Repeat over several gene folds and report mean ± spread.

*Practical notes:* fitting on a gene subset means re-running `parafac2_nd` on a sliced AnnData —
make sure `var["means"]` is recomputed for the slice, or centering will be inconsistent between
the fit and your manual least-squares step. Verify your implementation by checking that at the
fitted rank your manual reconstruction of the *training* block reproduces the R2X that
`parafac2_nd` reports.

### Approach 3 — Permutation null for R2X (parallel analysis)

**Goal:** turn a smooth monotone R2X curve into a curve with a crossing point, and give absolute
values meaning.

Refit RISE at every rank on data whose cross-gene structure is destroyed but whose marginals are
preserved: independently permute each gene's values across cells, within each sample. (Consider
also a second, weaker null that permutes cells across samples only, and report the difference —
the two nulls bracket "no structure at all" versus "no *sample-specific* structure.")

Then:
- Plot observed R2X and null R2X versus rank on the same axes.
- Plot **marginal** ΔR2X per added component for observed and null.
- The recommended rank is the largest `r` at which observed ΔR2X still exceeds the null's ΔR2X
  (i.e. the last component that explains more than a permuted component would).

This is cheap in code and expensive only in compute (one full rank sweep per null replicate).
Two or three null replicates is enough; the null curve is typically very low-variance. This also
supplies context that the paper currently lacks: with R2X ≈ 0.05, a reader needs to know the
noise floor is, say, 0.01 before the number means anything.

*Watch out:* permuting a sparse count matrix within a gene preserves the gene's marginal exactly
and is the right null. Do **not** permute the whole matrix, which would destroy the depth/library
size structure and produce an artificially low floor.

### Approach 4 — Count of replicable components, not mean FMS

**Goal:** replace an average that must decay smoothly with a count that plateaus.

`calculate_fms_rise` returns a single scalar averaged over matched components (with
`skip_mode=1`, correctly skipping the eigen-state mode, which is not comparable across fits).
Averaging guarantees smooth decay: at higher rank you average a few well-recovered components
against progressively more marginal ones.

Instead, from the same paired fits, compute the **per-component** matched similarity and report
`#{components with similarity > τ}` for τ ∈ {0.6, 0.7, 0.8} versus rank. Expect a rise then a
plateau; the plateau onset is the recommendation, and it supports the sentence "ranks above N
add components that do not reproduce."

Two refinements to test:
- **Match on the gene factor `C`.** It is the interpretable, cross-fit-comparable mode. Verify
  whether `tlviz.factor_tools.factor_match_score` can be persuaded to return per-component
  scores; if not, do the matching yourself with `scipy.optimize.linear_sum_assignment` on the
  absolute cosine-similarity matrix between the two `C` matrices (remember the sign ambiguity —
  use `|cos|`).
- **Split-half over samples, not bootstrap over cells.** The current `resample` bootstraps cells
  within each sample, which tests robustness to sequencing depth. Splitting *patients* into two
  halves and fitting each independently tests whether a program replicates across people, which
  is the claim the paper actually makes. Report both; if they disagree, that disagreement is
  itself a finding. (COVID has only 12 samples, so split-half is noisy — use many random splits.)

### Approach 5 — Subspace stability (principal angles)

**Goal:** a stability metric immune to component splitting.

FMS penalizes *splitting*: when a real program divides into two at higher rank, the match score
collapses even though nothing was learned wrongly. Instead compare the **subspaces** rather than
individual components:

- Between two fits (split-half or bootstrap), compute the principal angles between the column
  spaces of the two gene-loading matrices `C₁`, `C₂` (via SVD of `Q₁ᵀQ₂` where `Q` are
  orthonormalized bases). Report the mean cosine of principal angles, or the Grassmann distance.
- Optionally do the same on the per-cell projection coordinates for shared cells.
- Also compute the *nested* version: how much of the rank-`r` subspace is contained in the
  rank-`r+1` subspace fit to the same data? Sharp drops indicate the model reorganizing rather
  than refining.

Subspace agreement stays high while the added dimensions carry real signal and falls when they
are noise, and it falls more sharply than FMS. **Where approach 5 and approach 4 diverge is
informative:** high subspace agreement with low FMS means the rank is inflated by splitting, not
by noise, and that is a different (milder) problem.

### Approach 6 — Degeneracy and conditioning diagnostics

**Goal:** cheap, resampling-free upper bound on usable rank.

At each rank, from a single fit, report:
- Maximum pairwise `|cosine|` between columns of the gene factor `C` — the rank at which
  components begin duplicating each other.
- Condition number of `B` and of `C`.
- The spread of the component weights (how many components carry non-negligible weight).
- Run 3–5 random initializations (`random_state`) at each rank and report whether they converge
  to the same solution (via approach 5's subspace metric) and whether final R2X is
  init-dependent. Init-dependence appearing at rank `r` is a hard ceiling, and it also answers
  Reviewer #2's request for random-seed sensitivity.

This costs almost nothing on top of sweeps you are already running — collect it in the same loop.

---

## 4. Datasets and compute

Run every criterion on **both** COVID-19 BAL (`import_balf_covid`) and ALAD (`import_alad`).
Reviewer #2 specifically objected that different metrics were used per dataset, so the
recommended criterion must be shown to work identically on both.

Current published choices: 35 RISE components (COVID) and the ALAD rank selected by predictive
performance — check `cellcommunicationpf2/factors.py` and the figure modules for exact values
and reproduce them. Your sweeps must span a range that brackets these, e.g. ranks
{5, 10, 15, 20, 25, 30, 35, 40, 45, 50}; the existing cached sweep only goes to 19, which is
below the published choice and therefore cannot support it.

Compute discipline:
- Coarse rank grid first, refine only near any candidate optimum.
- Cache every sweep to CSV under `cellcommunicationpf2/data/` via `load_or_compute_dataframe`,
  following the naming pattern of the existing cached files, so nothing is recomputed.
- Prototype on COVID (smaller) before touching ALAD.
- Put scratch scripts and intermediate output in the scratchpad directory, not in the repo.
- If a single configuration is going to take more than ~30 minutes, say so in the report rather
  than silently running it.

---

## 5. How to decide which criterion wins

Judge the candidates on these criteria, in this order:

1. **Does it have an interior optimum or a well-defined cut point?** A monotone curve is a
   failed criterion regardless of how principled it is.
2. **Does it give a consistent answer on both datasets** using identical procedure and
   thresholds? A criterion needing per-dataset tuning does not answer Reviewer #2.
3. **Is it stable across replicates/seeds?** Report the spread, not just the mean.
4. **Is the recommended rank compatible with the published choices?** If the best criterion says
   the published rank is wrong, that is a legitimate and important finding — report it plainly
   with the evidence, and estimate what would have to be refit. Do not tune the criterion until
   it agrees with the published number.
5. **Is it explainable in three sentences of a response letter?** Prefer a simple criterion with
   a clear optimum over an elaborate one with a marginal one.
6. **Compute cost**, as a tiebreaker.

---

## 6. Deliverables

Write to the repository root unless told otherwise:

1. **`RISE_RANK_REPORT.md`** containing:
   - A one-paragraph recommendation up front: which criterion, what rank it selects on each
     dataset, and how confident you are.
   - One section per approach: what you implemented, what you found, plots referenced by path,
     and an explicit verdict (useful / not useful / inconclusive, and why).
   - The empirical answer to the approach-1a monotonicity question, stated explicitly.
   - A table of recommended rank per criterion per dataset.
   - Honest statement of any criterion that failed to discriminate.
   - A short recommendation on the accompanying **sensitivity analysis** (e.g. match gene
     factors `C` across ranks 25/35/45 and show the top-loading genes and the resulting
     eigen-states are stable) — the response letter will need this regardless of which
     criterion wins.
   - A draft paragraph, ~150 words, suitable for adaptation into the response to Reviewer #2
     major comment 4 (RISE-rank portion only).
2. **Reusable code** in `cellcommunicationpf2/` (new module, e.g. `rank_selection.py`) with the
   metrics implemented as tested functions, following existing style. Add pytest coverage for
   the non-obvious pieces — in particular a synthetic test that the bi-cross-validation error of
   approach 1b really does rise past the true rank on data simulated from a known PARAFAC2 model.
   That test is the main guard against a silently-leaking implementation.
3. **Cached sweep CSVs** under `cellcommunicationpf2/data/`.
4. **Draft figure module(s)** under `cellcommunicationpf2/figures/` for whichever criteria you
   recommend, in the existing `makeFigure()` idiom. Do not renumber or overwrite existing
   figures; use a provisional name.

Do **not** edit `manuscript/review-response.qmd` or any manuscript source. Do not commit.

---

## 7. Things to be careful about

- **Leakage** is the central technical risk. For any cross-validation scheme, write down
  explicitly which parameters were fit on which cells and which genes before trusting a curve.
  If you cannot write that down, the scheme is wrong.
- **Sign and permutation ambiguity**: PARAFAC2 components come out in arbitrary sign and order.
  `standardize_pf2` imposes an ordering, but it is data-dependent, so never compare components
  positionally across fits — always match, and always on `|cos|`.
- **Centering**: `anndata_to_list` subtracts `var["means"]`. Any manual reconstruction error must
  use the same centering, and any gene-subset refit must recompute means consistently.
- **Rank-vs-`B`-shape**: `B` is (rank × rank) here; do not assume it is (n_cells × rank).
- Bootstrap FMS in the existing cached data shows occasional catastrophic outliers (rank 4, run
  0 at 0.75 while sibling runs exceed 0.995). Determine whether these are convergence failures
  rather than genuine instability — check iteration counts and final error — before treating
  them as signal. If they are convergence failures, that alone may explain much of the noise in
  the current FMS figure, and tightening `tol`/`n_iter_max` may sharpen it for free.
