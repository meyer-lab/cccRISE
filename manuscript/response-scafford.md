# Point-by-Point Response Scaffold — MSB-2026-13900

Working scaffold for the rebuttal letter. Every numbered reviewer comment maps to a
planned action, the new/changed figure that answers it, and the manuscript section that
changes. Part references point to `revision-plan.md`. Status: ☐ not started · ◐ in
progress · ☑ done.

---

## Reviewer #1

| # | Concern (short) | Planned action | Fig / panel | Section | Part | Status |
|---|---|---|---|---|---|---|
| 1 | No comparison to any published method | Run Tensor-cell2cell, Scriabin, STACCato, NICHES/DeepCOLOR head-to-head; report ALAD AUC; single-condition degradation test vs Scriabin | New benchmark table + concordance fig | Results (new subsection), Methods | 2 | ☐ |
| 2 | Central "resolution" claim not quantitatively supported | SERGIO-based ground-truth simulation; precision/recall/F1/AUC for program + cell-driver recovery; concordance vs Scriabin on real data | New "Simulation validation" fig | Results, Methods | 1 (+2) | ☐ |
| 3 | Sign ambiguity in communication score (neg×neg → spurious positive) | Diagnose sign composition; enforce non-negativity + non-negative CPD; document sign convention; show components not sign artifacts | New audit panel | Methods (scoring), Results | 3 | ☐ |
| 4 | Overfitting risk — rank selection not nested in CV | Refactor to nested CV; report honest outer-fold AUC | Updated Fig 4b–d | Methods (logreg), Results | 4 | ☐ |
| 5 | Small samples, no independent classifier validation | State n=12 / no external cohort prominently; foreground negative CLAD/death result; use Fig S9 independent cohort | Text; Fig S9 | Results, Discussion | 4 | ☐ |
| 6 | No significance testing for communication programs | Add permutation null (within-condition, `compute_communication_significance`) + cross-condition differential null (`compute_differential_significance`, scDiffCom-analogous); empirical per-component significance | New observed-vs-null panel | Methods, Results | 4 | ◐ |
| 7 | Missing differential-CCC baselines (scDiffCom, CrossTalkeR) | Add to intro/discussion + references; link both permutation nulls (within-condition and cross-condition) to scDiffCom/CrossTalkeR's own significance schemes | — | Intro, Discussion, Refs | 6 | ◐ |
| 8 | "Label-free" premise vs label-dependent interpretation | Soften framing OR demonstrate ≥1 program identified de novo (marker enrichment, no prior labels) | New/updated panel | Abstract, Results, Discussion | 6 | ☐ |

## Reviewer #2 — Major

| # | Concern (short) | Planned action | Fig / panel | Section | Part | Status |
|---|---|---|---|---|---|---|
| 1 | No quantitative accuracy; recovers different programs than pseudobulk | Simulation with ground truth + competitor comparison (incl. Tensor-cell2cell); quantify full CCC-RISE↔pseudobulk component overlap, explain non-recovered | Simulation fig + overlap fig | Results, Methods | 1, 2 | ☐ |
| 2 | Components may be artifacts — no systematic biological validation | GO/pathway enrichment for top LR pairs per component and for logreg genes (Fig 4q/r/w/x, 5g/h) | New enrichment EV fig + tables | Results, Methods | 5 | ☐ |
| 3 | Score is a simple product with no statistical test | Permutation null (Part 4) + robustness to CellChat/CellPhoneDB-style scoring | Null panel + robustness panel | Methods, Results | 3, 4 | ☐ |
| 4 | Hyperparameter selection unclear; FMS described but unused; no sensitivity | Sharper consistent rank criterion; rank + seed sensitivity sweep with FMS; show conclusions persist | New sensitivity EV fig | Methods, Results | 6 | ☐ |
| 5 | Inconsistent condition-component selection across datasets | Apply one selection procedure to both datasets; reconcile with raw condition factor matrix | Updated Fig 3a / 4d, S5h | Results, Methods | 6 | ☐ |
| 6 | Inconsistent math notation; illogical Methods flow | Rewrite Methods in Fig 1 order (RISE → tensor → CPD); single consistent symbol table with biological glosses | — | Methods | 6 | ☐ |

## Reviewer #2 — Minor

| # | Concern (short) | Planned action | Fig / panel | Section | Part | Status |
|---|---|---|---|---|---|---|
| 1 | Why eigen-states vs CCA/PCA/NMF/VAE? | Justify PARAFAC2 (cross-condition alignment); note non-negative variant; brief comparison note | — | Discussion / Methods | 3, 6 | ☐ |
| 2 | Batch effects / patient heterogeneity handling | Explain RISE+CPD absorption of per-patient/technical variation; supplementary check | New supp panel | Methods | 7 | ☐ |
| 3 | Interpreting negative LR loadings; why not non-negative CPD | Adopt non-negative CPD path; explain interpretation | Tied to Part 3 | Methods, Results | 3 | ☐ |
| 4 | Logistic regression not fully described | Specify features (condition factor rows) and outcomes (ALAD status / stability) | — | Methods | 7 | ☐ |
| 5 | Colorbars must indicate normalization (Fig 2b,2i,2m) | Label normalization; describe method in legends | Fig 2b/i/m + others | Figure legends | 7 | ☐ |
| 6 | How was PaCMAP performed? | Add PaCMAP procedure + parameters | — | Methods | 7 | ☐ |
| 7 | Fig 1d mislabeled "sender Eigen-state Factor" | Correct label | Fig 1d | Figure | 7 | ☐ |

## Cross-comments

| Source | Concern (short) | Planned action | Part | Status |
|---|---|---|---|---|
| R1 cross | Analyses too limited (esp. quantitative); need synthetic experiments to quantify precision/recall across plausible range | Full SERGIO sweep across effect size, subpop fraction, #conditions, noise | 1 | ☐ |
| R2 cross | Provide exact mathematical formulation, consistent notation, biological explanations | Methods rewrite + symbol table | 6 | ☐ |

## Editorial (EMBO Press) checklist

| Item | Planned action | Status |
|---|---|---|
| .docx with tracked changes | Produce highlighted docx of manuscript | ☐ |
| Production-quality figure files (eps/tif/jpg, one per figure) | Export per guideline | ☐ |
| .docx response letter incl. reviewer reports + point-by-point | Assemble from this scaffold | ☐ |
| ORCID for corresponding author | Supply | ☐ |
| EV figures (≤5) cited as Figure EV1… + legends in main text | Reorganize supplement into EV + Appendix | ☐ |
| Appendix PDF (non-EV figs/tables) with ToC | Assemble | ☐ |
| Data deposition + Data Availability section (GEO/Zenodo accessions, code release tag) | Deposit + write section | ☐ |
| Source data for main figures | Provide per instructions | ☐ |
| Data citations in reference list ([DATASET]) | Add for reused datasets (Liao, Calabrese) | ☐ |
| Disclosure statement and competing interests (updated policy) | Add heading + review | ☐ |
| Structured Methods + Reagents and Tools Table (separate file) | Reformat Methods; fill template | ☐ |
| Stats in figure legends (n, test, error-bar definition, scale bars) | Audit every legend | ☐ |
| Standfirst (~250 char) + 3–4 bullet points + 550×300 visual abstract PNG | Produce | ☐ |
| Completed author checklist | Fill and include | ☐ |
