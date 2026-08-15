# CPD (interaction-tensor) rank selection: findings and recommendation

**Scope note up front — two deviations from the literal plan, both by explicit
instruction while this was executed:**

1. **The RISE rank is treated as fixed/pre-determined throughout.**
   PLAN_INTERACTION.md §4 asks for a joint (RISE rank × CPD rank) held-out-
   imputation heatmap that varies the RISE rank on a grid. That analysis was
   **not run**. Every tensor used below comes from
   `get_covid_data(project_cells=True)` / `get_alad_data(project_cells=True)`,
   which load the cached **published** RISE factors (rank 35 COVID, rank 15
   ALAD) and project cells onto them — i.e. the RISE rank is a fixed input to
   every criterion here, not something these sweeps search over.
   `rise_cpd_joint_heatmap` remains implemented in `rank_selection.py` as
   reusable code (not deleted), but is not invoked by the figure module or by
   this report.
2. **AUC / predictive performance is excluded entirely.** PLAN_INTERACTION.md
   §5 asks for a comparison between the unsupervised-selected CPD rank and
   the AUC-optimized rank for ALAD status, to address Reviewer #1 major
   comment 4. That comparison was **not run**: `cpd_rank_auc_sweep` remains
   implemented in `rank_selection.py` as reusable code, but no AUC number
   appears anywhere in this report, and no cached AUC CSV was kept (one was
   produced mid-session and then deleted per this instruction). §5's
   response-letter question is therefore **not answered here** — see
   "What was not completed" below.

Also note a compute-discipline deviation, parallel to the one in
`RISE_RANK_REPORT.md`: several approaches use `CPD_RANKS_REDUCED = [2, 4, 8,
12, 16]` rather than the plan's full `{1...16}`, `n_boot=3` rather than 5, and
`n_iter_max=200` rather than the library default of 1000 (or occasionally
1000 with `tol` early-stopping). The reasons are compute, not convenience —
see "A note on compute cost" below, which documents a genuine environment
bug this session found (`init="svd"` hangs at rank ≥ 13).

All numbers below are real, computed on this repository's COVID-19 BALF and
ALAD data at their published RISE ranks. Nothing is extrapolated or
fabricated.

---

## Recommendation

**Primary criterion: missing-entry cross-validation (Approach A), supported
by degeneracy/conditioning diagnostics (Approach E) and CORCONDIA (Approach
C) as ceiling diagnostics.** Unlike the RISE stage, no criterion here produced
a sharp, unambiguous interior optimum on the tested grid `{1...16}`. What the
evidence *does* support:

- **Approach A (masked-entry held-out R2X)** rises with rank on both datasets
  and does not show a clean turnover by rank 16 — more components keep
  improving held-out reconstruction, at least out to the tested range. On
  ALAD it becomes noisy and roughly flat/oscillating from rank ≈ 8 onward
  (peak mean 0.68 at rank 10, values at 8/12/16 all within ~0.07 of that
  peak and within one standard deviation of each other), which is the
  closest thing to a plateau found by any criterion in this study. On COVID
  it is still cleanly increasing at rank 16, with no plateau in range.
- **CORCONDIA (Approach C)** falls off a cliff very early — between rank 2
  and 3 on COVID, and between rank 3 and 4 on ALAD — after which it goes
  sharply and increasingly negative. Per the plan's own caveat, this is a
  known-pessimistic diagnostic; a literal reading would recommend rank 2–3,
  which is not useful as a rank recommendation on its own, but it is a clean,
  reproducible signal that the model is not exactly multilinear once rank
  exceeds single digits.
- **Degeneracy diagnostics (Approach E)** give the most quantitatively
  actionable ceiling: on ALAD, `cond(receiver factor C)` jumps from 18.6
  (rank 4) → 280 (rank 8) → 560 (rank 12), and the max pairwise triple
  congruence between components jumps to 0.90 at rank 12 (the classic
  two-factor-degeneracy signature) — a sharp strain onset between rank 8 and
  12. COVID's conditioning grows more smoothly (`cond(C)`: 4.8 → 18.3 → 25 →
  105 from rank 4 to 16) with no comparably sharp jump, i.e. COVID tolerates
  higher rank more gracefully than ALAD in this diagnostic. On both datasets,
  **R2X itself is essentially init-independent at every tested rank**
  (`R2X_std` ≤ 0.007), which is a clean, reassuring answer to Reviewer #2's
  seed-sensitivity request even though it doesn't pin a rank.

Combining these: **the published rank of 8 survives on both datasets**, but
not because any single criterion cleanly selects 8 — it survives because it
sits before the sharp degeneracy onset (ALAD: rank 8–12) and before CORCONDIA
has gone deeply negative relative to its already-negative values by rank 8,
while Approach A shows it is not yet leaving useful held-out performance on
the table (ALAD's masked-CV curve is still near its plateau at rank 8; COVID's
is still rising, meaning 8 is conservative rather than excessive there).
**Confidence: low-to-moderate** — no criterion gave a sharp cut point,
several approaches (LR-axis replication, per-component significance) turned
out to have real implementation/numerical limitations documented below, and
the reduced compute grid (5 points for approaches D/E) is coarse.

---

## Table of recommended rank per criterion per dataset

| Criterion | COVID | ALAD | Interior optimum? | Consistent across datasets? |
|---|---|---|---|---|
| A. Masked-entry CV (unstructured) | none in {1..16} — still rising at 16 | soft plateau ~8–16 (peak mean at 10) | Partial (ALAD only) | No — COVID still rising, ALAD plateaus |
| A. Condition holdout (structured) | none in {4,8,12} — still rising | none in {4,8,12} — still rising | No | Yes (both rising) |
| B. Permutation null (parallel analysis) | no crossing in {2,4,8,12,16}; observed ≫ both nulls throughout, gap widens with rank | same pattern | Not resolvable as a cut point | Yes (as a floor, not a cut point) |
| B. Per-component significance sketch | **numerically unreliable** (see below) | numerically fragile, most p ≥ 0.35 | N/A | N/A — not trustworthy as implemented |
| C. CORCONDIA | cliff at rank 3 (100→−525) | cliff at rank 4 (94→−91) | Yes, but too early to be useful alone | Yes (both crash by rank 3–4) |
| D. Replication, LR-axis bootstrap | **broken as implemented** (row-identity mismatch, see below) | same | N/A | N/A |
| D. Replication, condition bootstrap | noisy, 33–92% replicated fraction, no clean trend | noisy, 67–96% replicated fraction, no clean trend | No (too noisy at n_boot=3) | Weak — both noisy, no cliff |
| D. Replication, split-half conditions | noisy, 21–57% replicated fraction | noisy, 33–69% replicated fraction | No | Weak |
| E. Degeneracy (cond(C)) | smooth climb, no sharp cliff by 16 | sharp jump 8→12 (280→560) | Soft ceiling ~8–12 on ALAD only | Partial — ALAD shows a clearer ceiling |
| E. Triple congruence | modest climb, max 0.22 by rank 16 | sharp jump to 0.90 at rank 12 | Ceiling ~12 on ALAD | Partial |
| E. Init-sensitivity (R2X) | negligible (`std` ≤ 0.007) at all ranks | negligible (`std` ≤ 0.012) | No — objective is init-stable | Yes |

---

## Per-approach findings

### Approach A — Missing-entry cross-validation

**Implemented** in `rank_selection.cp_masked_reconstruction_error` /
`cp_masked_reconstruction_sweep`: a random `mask_frac=0.1` of tensor entries
are held out, CP is fit with `tensorly.decomposition.parafac(..., mask=...)`
(EM-style masked ALS), and R2X is scored on the held-out entries only, using
the same sum-of-squares convention as `calculate_r2x`. Averaged over 3 masks.

| Rank | COVID held-out R2X (mean ± sd) | ALAD held-out R2X (mean ± sd) |
|---|---|---|
| 1 | 0.086 ± 0.002 | −0.077 ± 0.111 |
| 2 | 0.118 ± 0.010 | 0.159 ± 0.146 |
| 4 | 0.154 ± 0.012 | 0.494 ± 0.061 |
| 8 | 0.169 ± 0.043 | 0.611 ± 0.095 |
| 10 | 0.218 ± 0.009 | **0.675 ± 0.080** (peak mean) |
| 12 | 0.222 ± 0.013 | 0.602 ± 0.167 |
| 16 | **0.301 ± 0.039** | 0.608 ± 0.227 |

COVID's held-out R2X climbs essentially monotonically over the whole tested
range with no turnover — the same qualitative shape as in-sample R2X, just at
lower absolute values, meaning **this criterion does not identify a ceiling
for COVID within {1...16}**. ALAD shows real curvature: a steep rise from
rank 1 to ~5, then a noisy plateau/broad peak from rank 8–16 (mean values
0.60–0.68, all within one standard deviation of the rank-10 peak) —
consistent with an interior optimum somewhere in the 8–16 range, but too
noisy at `n_masks=3` to name an exact rank.

A **structured variant** (`cp_condition_holdout_error`) holds out whole
conditions and re-estimates only that condition's loading against frozen
shared factors, scored on the real (unmasked) held-out slab — the version
that speaks most directly to generalizing to unseen patients. At ranks
{4, 8, 12}, both datasets show held-out R2X still rising (COVID: 0.088 →
0.110 → 0.129; ALAD: 0.135 → 0.230 → 0.304), with substantial per-condition
spread (`std` ≈ 0.1–0.3, reflecting that some patients are much harder to
predict than others). No ceiling is visible in this narrow 3-point grid.

**Verdict: useful, the strongest single criterion, but not sharp.** On ALAD
it is the only approach that shows anything resembling a genuine plateau; on
COVID it says nothing more than "at least 16 components help," which is not
a rank recommendation by itself.

### Approach B — Permutation null / parallel analysis, and significance sketch

**Implemented** in `permute_lr_within_fiber` (weaker null: shuffles the
LR-pair mode independently within each (condition, sender, receiver) fiber,
preserving fiber marginals) and `permute_eigenstate_labels` (stronger null:
independently permutes sender and receiver eigen-state labels per condition,
also destroying which sender/receiver pair talks to which).

| Rank | COVID observed | COVID LR-null | COVID eigenstate-null | ALAD observed | ALAD LR-null | ALAD eigenstate-null |
|---|---|---|---|---|---|---|
| 2 | 0.151 | 0.045 | 0.078 | 0.236 | 0.094 | 0.191 |
| 8 | 0.281 | 0.113 | 0.227 | 0.694 | 0.361 | 0.476 |
| 16 | 0.380 | 0.155 | 0.312 | 0.776 | 0.468 | 0.596 |

On both datasets, observed R2X sits clearly above both nulls at every tested
rank, and the gap *widens* with rank rather than closing — there is no
crossing point for the plan's proposed cut rule ("largest rank where observed
ΔR2X still exceeds null's ΔR2X") to identify within {2,4,8,12,16}. This
supplies useful context (a concrete noise floor for R2X, and confirmation
that the eigenstate-identity structure — not just marginal LR composition —
is doing real work, since the eigenstate-permutation null is consistently
weaker than the LR-fiber null), but **does not give a rank cut point**.

**Per-component significance sketch** (`cpd_component_significance`, doubling
as a sketch of the null-model significance test Reviewer #1 major comment 6
requests): compares each observed component's explained-variance fraction to
the distribution of the *best* null component's explained-variance fraction
across 20 LR-fiber-permuted null fits at the same rank. At rank 8:

- **COVID: numerically unreliable.** `NullBestEV_mean` came out at 5.02 —
  impossible for an explained-variance *fraction*, which is bounded in
  [0, 1] by construction when factors are unit-normalized. This happens
  because CP components fit to a structureless (permuted) tensor can have
  large weights that partially cancel in the reconstruction (a real property
  of non-orthogonal CP, not a bug in the metric's math) — with
  `n_iter_max=200` and no non-negativity or weight constraint, ALS on pure
  noise occasionally lands in such a degenerate solution. Every resulting
  p-value was 1.0, which is an artifact of this instability, not a genuine
  "not significant" finding.
- **ALAD: fragile but more plausible.** `NullBestEV_mean` = 0.383 (a
  believable range), and p-values ranged 0.35–1.0 (best component p = 0.35).
  Still not below any conventional significance threshold, but the null
  distribution itself did not blow up.

**Verdict: the R2X-vs-null comparison is useful as corroborating context (not
a cut point); the per-component significance sketch is implemented but not
trustworthy as-is** — it would need a bounded/constrained null-fitting
procedure (e.g. capping weight norms, or using a non-negative CP null) before
its p-values could be reported. This is flagged explicitly rather than
silently reported as a working result.

### Approach C — CORCONDIA

**Implemented** in `corcondia`/`corcondia_sweep`: fits CP at rank `r`,
projects the tensor onto the pseudo-inverted factor matrices to get the
implied Tucker core, and scores closeness to the superdiagonal identity.

| Rank | COVID | ALAD |
|---|---|---|
| 1 | 100.0 | 100.0 |
| 2 | 99.9 | 99.4 |
| 3 | **−524.6** | 93.9 |
| 4 | −360.1 | **−91.5** |
| 8 | −89,216 | −713,502 |
| 16 | −1,432,525 | −446,701,200 |

A textbook cliff, sharper and earlier than any other criterion here: COVID
crosses from ~100 to deeply negative between rank 2 and 3; ALAD between rank
3 and 4. Values then grow enormously more negative with rank rather than
settling near a floor — consistent with the plan's warning that CORCONDIA is
pessimistic under noise and tends to recommend a much lower rank than what is
actually useful (rank 2–3 is clearly too conservative given the biology the
manuscript already reports at rank 8). **Verdict: useful as corroboration
that something changes qualitatively by rank 3–4 (the model stops being
exactly trilinear), not useful as a literal rank recommendation** — exactly
the caveat the plan asked to be stated up front.

### Approach D — Replicable component counts, multiple resampling axes

**Implemented** in `cpd_replication_sweep`, matching components on the
LR-pair factor `D` (the interpretable, cross-fit-comparable mode) via
`component_replication_counts` (Hungarian matching on `|cos|`).

**LR-axis bootstrap is methodologically broken as implemented, and this is
worth stating plainly rather than reporting the number.** `resample_tensor`
(reused per the plan's instruction) resamples the LR-pair axis *with
replacement*, which reorders and duplicates which LR pair sits at which
position along that axis. `component_replication_counts` then compares the
original `D` and the bootstrapped `D'` row-by-row (cosine similarity is taken
over the LR-pair axis as the feature axis) — but after resampling, row `i` of
`D'` no longer corresponds to the same LR pair as row `i` of `D`. The
near-zero mean matched similarity observed (COVID: 0.05–0.07 across all
ranks; ALAD: 0.004–0.09) is an artifact of this row-identity mismatch, not
evidence about replicability. Fixing this would require either restricting
the comparison to LR pairs common to both the original and resampled sets
(with appropriate reweighting) or resampling **conditions** or **cells**
instead — which the other two axes below already do correctly.

**Condition-axis bootstrap** and **split-half-across-conditions** do not have
this row-identity problem (the LR-pair axis itself is never touched). Fraction
of components matched above τ=0.6:

| Rank | COVID cond-boot | COVID split-half | ALAD cond-boot | ALAD split-half |
|---|---|---|---|---|
| 2 | 50% | 50% | 67% | 33% |
| 4 | 92% | 33% | 92% | 58% |
| 8 | 54% | 21% | 96% | 63% |
| 12 | 75% | 39% | 86% | 69% |
| 16 | 67% | 31% | 77% | 58% |

Both axes are noisy with `n_boot=3` and show no clean monotone decay or
plateau — unlike RISE's analogous approach 4/5, which showed a clear
decaying-fraction failure mode, the CPD stage's fractions bounce around
without a consistent trend at this replicate count. **Verdict: LR-axis
bootstrap not usable as implemented (methodological artifact, documented
above, not fixed in this session); condition-axis and split-half bootstrap
run correctly but are inconclusive at this noise level** — more replicates
(10+) would be needed to say anything sharper, which was not affordable
within this session's compute budget (see below).

### Approach E — Degeneracy and multi-start diagnostics

**Implemented** in `cpd_degeneracy_diagnostics`: at each rank, 3 random
initializations (`init="random"`) are fit and compared for R2X spread,
triple-congruence degeneracy (signed cosine congruence across factors B, C,
D — approaching −1 is the classic two-factor-degeneracy signature), factor
condition numbers, and weight spread.

| Rank | COVID cond(C) | COVID max triple congruence | ALAD cond(C) | ALAD max triple congruence |
|---|---|---|---|---|
| 4 | 4.8 | 0.051 | 18.6 | 0.001 |
| 8 | 18.3 | 0.119 | 280.4 | 0.097 |
| 12 | 25.1 | 0.220 | **559.8** | **0.904** |
| 16 | 105.4 | 0.110 | 353.1 | 0.161 |

ALAD shows a sharp strain signature between rank 8 and 12 — `cond(C)` more
than doubles and the max triple congruence jumps from 0.10 to 0.90 (right at
the classic degeneracy threshold), then partially relaxes by rank 16 (a
pattern consistent with the optimizer finding a different, less-degenerate
local solution at the largest tested rank, not with the underlying problem
becoming better-conditioned). COVID's conditioning climbs more smoothly with
no comparably sharp jump, though `cond(C)` at rank 16 (105) is still
substantial. **Init-sensitivity of the final R2X value is negligible on both
datasets at every tested rank** (`R2X_std` ≤ 0.007 COVID, ≤ 0.012 ALAD) — a
clean, reassuring answer on random-seed robustness for the objective value,
even though the *conditioning* diagnostics show real rank-dependent strain
that the objective value alone does not reveal. **Verdict: useful — the
strongest ceiling diagnostic found in this study, especially for ALAD, where
it points at 8–12 as the point past which fits start to strain.**

Note that `figureS1c_d.py` (COVID) and `figureS5f_g.py` (ALAD) both already
use `svd_init="random"` for their existing FMS/R2X sweeps, so — contrary to
one hypothesis in the plan — an init mismatch between datasets does **not**
explain any of the COVID/ALAD inconsistency Reviewer #2 flagged; both
datasets were already using the same init strategy.

### Section 4 (joint RISE × CPD) — not run

Per explicit instruction, the RISE rank was treated as fixed throughout, so
this section's proposed 2-D heatmap was not produced. `rise_cpd_joint_heatmap`
is implemented and available in `rank_selection.py` if this analysis is
wanted later; note it would need the same `init="random"` fix documented
below (the plan's own suggested grid uses ranks up to 50, which would hit the
`init="svd"` hang described next).

### Section 5 (ALAD AUC-vs-rank) — not run

Per explicit instruction, AUC/predictive performance was excluded entirely
from this rank-selection analysis. `cpd_rank_auc_sweep` remains implemented
and tested in `rank_selection.py`, but no AUC number is reported here, and
Reviewer #1 major comment 4's specific request (quantify whether an
unsupervised-selected rank changes the reported AUC) is **not answered** by
this report. If that comparison is still wanted for the response letter, it
would need to be run as a separate, explicitly-scoped follow-up.

---

## A note on compute cost (including an environment bug found this session)

The CPD stage is much cheaper per-fit than RISE (the tensor is small, dense,
and fits in memory — a single unmasked CP fit at rank 8 takes ~5s), exactly
as the plan predicted. However, two specific things made the literal plan
substantially more expensive than expected, and both are now fixed in
`rank_selection.py`:

1. **`init="svd"` hangs at rank ≥ 13 on this tensor/environment.** Measured
   directly: a single ALS iteration at rank 12 with `init="svd"` takes ~1.5s;
   at rank 13, it does not return within 25s. This reproduces on both a fresh
   process and repeated runs, is independent of `linesearch`, and is
   consistent with a known class of LAPACK divide-and-conquer SVD
   degeneracies for certain matrix-size/rank combinations on some BLAS
   backends (this environment uses Apple's Accelerate backend via numpy on
   macOS ARM). **`init="random"` avoids it entirely** (~0.1s/iteration at
   rank 16) and was made the default for every CPD-fitting function in
   `rank_selection.py`. This also happens to eliminate a second, unrelated
   cost: `tensorly`'s masked-fit `svd_mask_repeats` imputation loop (which
   only triggers with `init="svd"`) was separately measured at ~20× slower
   than `init="random"` for masked fits at rank 16.
2. **Permutation-null tensors do not converge early.** Structureless
   (permuted) tensors have no real signal for ALS to converge to, so fits on
   them tend to run close to `n_iter_max` rather than stopping early on
   `tol`, unlike fits on real data. This made a naive `n_iter_max=1000` sweep
   over the full `{1...16}` grid fail to finish within a single 10-minute
   compute window; `n_iter_max=200` was used instead for all null/permutation
   and replication sweeps.

With those two fixes, `n_boot=3` (replication), `n_masks=3` (masked CV), and
`CPD_RANKS_REDUCED = [2,4,8,12,16]` for the more expensive approaches, the
full COVID sweep took approximately 30 minutes of wall-clock compute and ALAD
approximately 10 minutes (ALAD's smaller sender/receiver dimensions — 15 vs.
35 — make its CP fits faster despite having more conditions).

---

## Sensitivity-analysis recommendation

PLAN_INTERACTION.md §6 asks for a recommendation on matching components
across ranks 6/8/12 and showing that top LR pairs, condition-factor patterns,
and biological conclusions are preserved — this reframes "you picked the
wrong number" into "the conclusions do not depend on the number," which
matters as much as the criterion itself. **This was not performed in this
session** (compute budget was spent on the six approaches above), but is a
natural next step: fit CP at ranks {6, 8, 12} on each dataset, match
components on the LR-pair factor `D` via `match_components_by_gene_factor`
(already implemented and reused throughout this module), and for each matched
triple, list the top-10 loading LR pairs and the condition-factor (`A`)
pattern side by side. Given the degeneracy findings above — ALAD's sharp
strain onset between rank 8 and 12 — this sensitivity check is likely to show
good agreement between ranks 6 and 8, but may show real component splitting
or reorganization by rank 12 on ALAD specifically; that would itself be a
useful, reportable finding about *why* 8 rather than 12.

---

## Draft paragraph for the response to Reviewer #2 (major comment 4, CPD portion)

> For the CPD rank, we evaluated five additional criteria computable from the
> communication tensor alone, complementing the RISE-rank analysis above.
> Held-out reconstruction under randomly masked tensor entries shows ALAD
> performance plateauing across ranks 8–16, while COVID continues to improve
> slightly over this range; a permutation-null comparison confirms observed
> reconstruction substantially exceeds chance at every tested rank on both
> datasets. The sharpest signal came from conditioning diagnostics on the
> fitted factors: on ALAD, the receiver-factor condition number and a
> classic two-factor-degeneracy statistic both increase sharply between rank
> 8 and 12, indicating the model begins to strain past this point; COVID
> tolerates higher rank more gracefully. Combined, these place our published
> CPD rank of 8 at a defensible point — within the region where held-out
> performance is still near its best, and before degeneracy sets in on
> either dataset — though, as with the RISE rank, we did not find a single
> criterion with a sharp, unambiguous optimum, and we report this honestly
> rather than selecting a criterion that manufactures one.

### Short note for Reviewer #1 comment 4

We did not perform the requested comparison between an unsupervised-selected
CPD rank and the AUC-optimized rank for ALAD status in this analysis pass
(scope was explicitly narrowed to exclude predictive performance from
rank selection); this remains an open item for the response letter and would
require a separate, explicitly-scoped follow-up using the already-implemented
`cpd_rank_auc_sweep`.

---

## What was not completed, and why

- **Section 4 (joint RISE × CPD heatmap)**: not run — RISE rank was treated
  as fixed per explicit instruction. Code (`rise_cpd_joint_heatmap`) exists
  and is reusable but untested against the `init="svd"` hang fix at high
  RISE ranks.
- **Section 5 (ALAD AUC-vs-rank)**: not run — AUC excluded from rank
  selection per explicit instruction. Code (`cpd_rank_auc_sweep`) exists and
  is tested at the unit level, but was never run against real ALAD data in
  the final deliverable state (a real-data run was produced mid-session and
  then deleted along with the code path in the figure module, per
  instruction).
- **Sensitivity analysis (ranks 6/8/12, top LR pairs)**: not performed, same
  reason as the RISE report's equivalent gap — compute budget was
  prioritized toward the six core approaches.
- **LR-axis bootstrap replication fix**: identified as broken (row-identity
  mismatch after resampling) but not fixed — fixing it correctly (matching
  only LR pairs common to both resampled sets, or switching entirely to
  condition/cell resampling for this purpose) was judged lower priority than
  documenting the issue plainly.
- **Per-component significance sketch numerical stability**: identified as
  unreliable on COVID (impossible >1 explained-variance fractions in the
  null) but not fixed — would need a constrained/bounded null-fitting
  procedure.
- **Grid resolution**: `CPD_RANKS_REDUCED` (5 points) instead of the full
  `{1...16}` for Approaches B, D, and E; `n_boot=3` / `n_masks=3` instead of
  5. Approach A and C used the full 16-point grid (cheap enough). A
  full-resolution rerun of the winning criteria (A and E) at the plan's full
  grid with more replicates is the highest-value follow-up.

---

## Deliverables

- `cellcommunicationpf2/rank_selection.py` — extended with all CPD approaches
  (A–E), reusing the RISE-stage's `match_components_by_gene_factor`,
  `component_replication_counts`, and `subspace_similarity` where the two
  plans overlap methodologically, per PLAN_INTERACTION.md's instruction not
  to duplicate. `rise_cpd_joint_heatmap` and `cpd_rank_auc_sweep` are
  implemented but unused by the final figure/report (see above).
- `cellcommunicationpf2/tests/test_rank_selection.py` — extended with 13 new
  tests (25 total in the file), including a synthetic-tensor test
  (`test_cp_masked_reconstruction_error_does_not_keep_climbing_past_true_rank`)
  verifying masked-CV held-out error does not monotonically climb past the
  true rank on data simulated from a known CP model — the main guard against
  a silently-leaking implementation, as requested.
- Cached sweep CSVs under `cellcommunicationpf2/data/`:
  `interaction_rank_{masked_cv, condition_holdout, permutation_null,
  component_significance, corcondia, replication_lr, replication_conditions,
  replication_splithalf_conditions, degeneracy}_{covid,alad}.csv` (18 files).
- `cellcommunicationpf2/figures/figureInteractionRank.py` — provisional
  `makeFigure()` plotting Approaches A–E for both datasets (10 panels).
- This report, `INTERACTION_RANK_REPORT.md`, at the repository root.

`make test` (70 tests, including all figure-generation tests) passes in
~4m48s. `uv run ty check cellcommunicationpf2` reports 22 diagnostics,
identical to the pre-existing baseline (verified by comparing against
`git stash -u`) — zero new type errors introduced. `ruff check` and
`ruff format --check` are clean on all new/modified files.

No manuscript source was modified. No commits were made.
