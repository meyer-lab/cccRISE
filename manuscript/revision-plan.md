# CCC-RISE Revision Plan — MSB-2026-13900

**Manuscript:** "Resolving cell-cell communication programs beyond cell-type
boundaries across conditions"
**Decision:** Major revision (single round; acceptance depends on completeness).

This document is the internal roadmap for the revision. It is organized into parts by
data / analysis / resource, maps each work item to the reviewer concern(s) it answers,
and flags the compute dependency. A companion point-by-point scaffold is in
`response-scafford.md`.

---

## Compute constraint (read first)

The RISE step — `parafac2_nd` from the `meyer-lab/parafac2` package (pinned in
`pyproject.toml`) — is the **GPU/CUDA-dependent** bottleneck and must run on the
cluster. Everything downstream of a cached RISE factorization is CPU-friendly: CPD
(`tensorly.parafac`), logistic regression, permutation nulls, GO enrichment,
simulation _scoring_, and figure generation. Each item below is tagged:

- **[GPU]** — requires the CUDA cluster (RISE fits / large sweeps).
- **[CPU]** — runnable on any workstation once RISE outputs are cached.

**Critical path:** the communication-score sign fix (Part 3) changes the tensor and can
shift downstream results, so it must be resolved _before_ re-running the simulation
benchmark, competitor comparisons, classifiers, and all figures.

---

## Triage: what decides the paper vs. what is goodwill

**CORE (decides acceptance):**

- Ground-truth simulation with precision/recall — R1#2, R2#1 (Part 1)
- Head-to-head competitor benchmarking incl. ALAD AUC — R1#1, R2#1 (Part 2)
- Communication-score sign fix + scoring robustness — R1#3, R2#3, R2-minor#3/#1 (Part 3)
- Nested-CV re-estimation of classifier AUC — R1#4/#5 (Part 4)
- Permutation null for component significance — R1#6, R2#3 (Part 4)

**SUBSTANTIVE (strengthens):**

- GO enrichment of components & logreg genes — R2#2 (Part 5)
- Unify condition-component selection across datasets — R2#5 (Part 6)
- Rank/seed sensitivity via FMS — R2#4 (Part 6)
- Methods rewrite with consistent notation — R2#6 (Part 6)
- Reconcile "label-free" framing — R1#8 (Part 6)
- Add missing differential-CCC methods — R1#7 (Part 6)

**CLARIFY / EDITORIAL (low effort):** non-negative-CPD rationale, batch effects, logreg
spec, colorbars/PaCMAP text, Fig 1d relabel, plus all EMBO production items (Part 7).

---

## Part 1 — Ground-truth simulation framework **[GPU for RISE; CPU for scoring]**

**Answers: R1#2, R2#1 (the crux). Highest impact.**

- Adopt a **SERGIO-based** simulator producing multi-condition scRNA-seq with **known**
  ligand→receptor programs and **known driver cells** that cut across nominal cell
  types.
- Encode ground truth at both levels CCC-RISE claims to resolve: (a) which LR programs
  vary with condition, (b) which individual cells drive each program.
- Run CCC-RISE and all competitors (Part 2) on identical simulations; report
  **precision, recall, F1, and AUC** for (i) program recovery and (ii) cell-level driver
  identification.
- Sweep a biologically plausible range — effect size, driver-subpopulation fraction,
  number of conditions, dropout/noise — as R1 requested in cross-commenting.
- **New code:** `cellcommunicationpf2/simulation.py` (generator + ground-truth registry
    - scoring metrics); `figures/figureSimX.py`.
- **New figure:** "Quantitative validation on simulated ground truth" — ROC/PR curves
  plus a recall-vs-subpopulation-fraction sweep heatmap.

## Part 2 — Competitor benchmarking **[GPU for RISE; separate envs for competitors]**

**Answers: R1#1, R2#1. Highest impact.**

- **Tensor-cell2cell:** run on COVID + ALAD; compare condition factors and **report
  ALAD-status AUC head-to-head**. Directly answers "can CCC-RISE recover what cell-type
  tensor methods find?"
- **Scriabin:** run condition-by-condition; compare per-cell communication to CCC-RISE
  per-cell loadings (concordance). Add a **single-condition** task to test whether
  CCC-RISE degrades vs. Scriabin when the multi-condition advantage is removed (R1#1).
- **STACCato:** ALAD-status AUC comparison (R1#1).
- **NICHES / DeepCOLOR:** additional single-cell-resolution concordance.
- **Pseudobulk-overlap concern (R2#1):** quantify overlap between _all_ CCC-RISE and
  pseudobulk-CPD components (not just the two highlighted), and classify each
  non-recovered pseudobulk component as recovered-but-reordered vs. genuinely absent —
  with data, not assertion.
- **Infrastructure:** a `benchmarks/` area with per-tool isolated environments (several
  are R / have conflicting deps) and cached outputs, kept out of the core package.
- **Deliverable:** benchmark table (AUC/accuracy × method × dataset) + concordance
  figure.

## Part 3 — Communication-score sign fix & robustness **[CPU once RISE cached]**

**Answers: R1#3, R2#3, R2-minor#3/#1. Fundamental — do early.**

- **Root cause:** `compute_ccc_matrix` (`cellcommunicationpf2/ccc.py:92`) takes
  `np.outer(prot_a_exp, prot_b_exp)` on RISE-projected eigen-state expression
  (`proj.T @ tensor`, `ccc_rise.py:896`), which is **not sign-constrained** →
  negative ligand × negative receptor gives a **spurious positive** score.
- **Diagnose:** audit the sign composition of the top LR contributions per component;
  quantify how many/how strongly reported components depend on negative×negative
  products.
- **Fix (recommend combining):** (a) enforce non-negativity on eigen-state expression
  before scoring; (b) use **non-negative CPD** on the communication tensor
  (`tensorly.non_negative_parafac`, replacing the mislabeled plain `parafac` at
  `ccc_rise.py:1078` and adding a non-negative path through `ccc_rise`/`tensor`). This
  also directly answers R2#1/#3 "why not non-negative CPD." Document the exact sign
  convention before/after the outer product.
- **Scoring robustness (R2#3):** re-score with CellChat- and CellPhoneDB-style scoring
  (extend `compute_ccc_matrix`, which already exposes `expression_mean`/`_gmean`; add a
  CellChat-style thresholded/Hill score) and show headline components are stable across
  scoring choices.

## Part 4 — Statistical rigor: nested CV + permutation null **[CPU once RISE cached]**

**Answers: R1#4, R1#5, R1#6, R2#3.**

- **Nested CV (R1#4):** Select rank of decomposition on reconstruction, not prediction.
- **Small-sample honesty (R1#5):** state plainly that COVID n=12 and the ALAD classifier
  lack an external validation cohort; foreground the existing negative result
  (components do not predict CLAD/death). Lean on the independent CD70/airway-brush
  cohort already in Fig S9 where possible.
- **Permutation null (R1#6, R2#3):** add a principled null — permute cell/condition
  labels (and/or LR identities) and recompute the communication statistic to assign
  empirical per-component significance. **New code:** `null_model.py`; **new panel:**
  observed vs. null. Supplies the CellPhoneDB/CellChat-style significance the reviewers
  ask for.

## Part 5 — Biological validation: GO enrichment **[CPU]**

**Answers: R2#2.**

- GO/pathway enrichment (`gseapy` or `decoupler`) on (a) top-weighted LR pairs per
  component and (b) logreg differentially-weighted genes (Fig 4q/r/w/x, Fig 5g/h).
- Show condition-specific components enrich for condition-relevant biology (antiviral /
  epithelial-repair in COVID; alloimmune / tolerance in ALAD) → programs are coherent
  biology, not decomposition artifacts.
- **Deliverable:** enrichment EV figure + supplementary tables.

## Part 6 — Consistency, sensitivity & Methods rewrite **[CPU; RISE sweeps GPU]**

**Answers: R2#4, R2#5, R2#6, R1#7, R1#8.**

- **Unify component selection (R2#5):** apply one selection procedure to both datasets
  (recommend logistic-regression-based, Spearman as secondary), and reconcile selected
  components against the raw condition factor matrix so severity/ALAD components are
  visibly enriched (addressing "everything looks enriched in ALAD-declined").
- **Sensitivity & seeds (R2#4):** sweep RISE rank (~25–50) and CPD rank (~6–12 COVID /
  ~12–24 ALAD) across **multiple seeds**; report FMS stability and whether biological
  conclusions persist (reuse `run_fms_r2x_analysis`, add seed looping). Provide a
  sharper, dataset-consistent rank-selection criterion (FMS elbow + R2X).
- **Methods rewrite (R2#6):** restructure to Fig 1 order — (1) RISE → eigen-states,
  (2) 4D communication-tensor construction, (3) CPD — with one **consistent symbol
  table** (current text reuses/collides symbols, e.g. $B$) and a biological gloss for
  every symbol.
- **Label-free framing (R1#8):** soften "beyond cell-type boundaries" _or_ demonstrate
  for ≥1 program that identity was established **without** reusing prior annotations
  (de novo marker enrichment on the driver subpopulation).
- **Differential-CCC methods (R1#7):** add scDiffCom and CrossTalkeR to intro/discussion
  and references; connect to the Part 4 permutation null.

## Part 7 — Minor comments & EMBO editorial **[CPU / non-code]**

**Answers: R2-minor#2/#4/#5/#6/#7 + editor's list.**

- **Batch effects (R2-minor#2):** explain whether/how RISE+CPD absorb per-patient /
  technical variation; add a supplementary check.
- **Logreg spec (R2-minor#4):** document input features (condition factor `uns["A"]`
  rows) and outcome (ALAD status / stability) in Methods.
- **Colorbars/normalization (R2-minor#5), PaCMAP description (R2-minor#6), Fig 1d
  relabel** "sender Eigen-state Factor" (R2-minor#7).
- **Editorial:** docx with tracked changes; EV figures (≤5) + Appendix packaging;
  Structured Methods + Reagents & Tools Table; **Data Availability** section (GEO/Zenodo
  accessions + code release tag); ORCID; standfirst + bullet points + 550×300 visual
  abstract; competing-interests heading; completed checklist; data citations in
  references.

---

## Suggested sequencing

1. **Part 3 (sign fix)** — unblocks everything; re-fit COVID + ALAD once resolved. [GPU]
2. **Parts 1 & 2 (simulation + competitors)** — the core quantitative story; run in
   parallel with the re-fit. [GPU + external envs]
3. **Part 4 (nested CV + null)** — on the re-fit outputs. [CPU]
4. **Part 5 (enrichment)** and **Part 6 (consistency/sensitivity/Methods)**. [CPU/GPU]
5. **Part 7 (minor + editorial)** — final assembly, docx, EV/Appendix, deposition.

## Files touched / added

- Edit: `cellcommunicationpf2/ccc.py` (scoring), `ccc_rise.py` (non-negative CPD path),
  `logreg.py` (nested CV), figure modules.
- Add: `cellcommunicationpf2/simulation.py`, `cellcommunicationpf2/null_model.py`,
  `benchmarks/` (competitor envs + cached outputs), new `figures/figureSim*.py` /
  enrichment / null panels.
- Manuscript: Methods rewrite, new/updated main + EV figures, Data Availability, and
  editorial deliverables.
