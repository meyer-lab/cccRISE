# Figure 5 results outline: cccRISE on the IBDverse terminal-ileum cohort

An outline for writing the Figure 5 results section. It is organized as aims, then sub-aims,
then the evidence for each: figure panels, our analyses, and the outside literature that
supports the biological reading.

Where a panel reuses a Figure 2 or Figure 3 analysis, the outline names the analogue, so the
method can be described the same way. Numbers are those in the built figures and their
source data (`uv run cccrise-fbuild 5`, `uv run cccrise-sourcedata 5`).

Panel letters follow `manuscript/_panels.yml`. The supplements are named here by their
module, `figureS_ibd_*`. Their EV or Appendix numbers are still to be assigned in
`_quarto.yml`.

## Setup

- **Data.** IBDverse terminal-ileum biopsies (Krzak et al. 2026, *Nat Genet*,
  doi:10.1038/s41588-026-02634-7).
  - 343 biopsies: 232 healthy, 47 Crohn's (CD) uninflamed, 64 CD inflamed.
  - At most 2,000 cells per biopsy; 634,534 cells.
  - The mesenchymal lineage is excluded as a biopsy-sampling artefact.
  - Every biopsy is from a different individual (343 individuals), so biopsies are the unit
    of every test.
- **Model.** RISE rank 60, CP rank 6 (`figureS_ibd_rank_selection`, chosen as Figure 2A–D
  chooses ALAD's). `figureS_ibd_rank_selection` G–H shows the RISE condition and gene
  factors as for ALAD.
- **Cohort.** `figureS_ibd_cohort` gives the schematic placeholder and the lineage and fine
  cell-type composition by status, as ALAD's cohort supplement does.
- **Per-cell weights.** Projections @ cp_B (sender) or cp_C (receiver), unscaled, as in
  Figure 2L/3.
- **Condition loadings.** Read-depth corrected (`utils.correct_by_read_depth`), as Figure 3
  tests them.
- **Tables.** The slow tables behind the program panels are written once by
  `cccrise-ibd-programs` into `cccrise/data/ibd_programs/`, which has its own
  `provenance.yaml`.
- **Two programs, two kinds of finding.**
  - The hormone programs (C3, C4) show cccRISE resolving functional states below the
    annotation.
  - The inflammatory programs (C1, C5) show cccRISE recovering a known regulatory circuit
    as coordinated programs.

## Aim 1. The six IBD programs and their disease associations

**1a. What the programs are (Fig 5A–D; `figureS_ibd_components` A–B).** This is Figure 2's
factor-matrix panel.
- D shows the top 2% of LR pairs (25 rows), because each IBD component is carried by only a
  handful of pairs.
- C3 is essentially PYY–NPY1R. C4 is GCG–GLP1R.
- C1 is IL1A/IL1B → IL1R2 plus IL-1 → IL1R1/IL1RAP and CCL → CCR1.
- C5 is CCL3L1–CCR1 and CXCL8/CXCL1 → CXCR1/2. IL1A–IL1R2 loads *negatively* on C5 (the
  blue cell in D).
- C2 is MAG–MAG (mast cells, ILCs). C6 is CDH4–CDH4.
- The mean per-cell weight per fine type is Figure 3's cell-type association heatmap
  (`figureS_ibd_components` A–B):
  - C3 senders are EE cells and its receivers EC cells;
  - C4 senders are EE cells and its receivers tuft cells;
  - C1 and C5 senders and receivers are SOD2+ monocytes and CCL3/4+ CXCL9/10+ macrophages.

**1b. Which programs track inflammation (`figureS_ibd_associations`; `figureS_ibd_components`
C).** This is Figure 3's association test and its ALAD supplement.
- **Method.** One-way ANOVA of each component's read-depth-corrected loading against three
  associations: 3-way status, healthy vs CD, and CD uninflamed vs inflamed. η² is the effect
  size, and p is BH-corrected across all associations × components. The RISE components get
  the same test.
- **CP components (A).**
  - 3-way: C1 0.05, C3 0.04, C4 0.03, C5 0.02 (all FDR < 0.05). C2 and C6 n.s.
  - Healthy vs CD: C1 0.04, C3 0.02, C5 0.02.
  - CD uninflamed vs inflamed: C3 0.09 and C4 0.09, both lower in inflamed biopsies.
- **Loadings (C–F).** C1 by 3-way status and healthy vs CD; C3 and C4, CD uninflamed vs
  inflamed.
- **RISE components (B).** 48 of 60 are FDR-significant for 3-way status.
- **Covariate screen (G)**, uncorrected, as for ALAD.
  - Chip, age and the lineage fractions track status: plasma B cell η² 0.26 and myeloid 0.24
    for 3-way status.
  - For CD uninflamed vs inflamed, read depth and the secretory and stem-cell fractions reach
    0.09, the same as C3/C4.
- **Loading concentration (`figureS_ibd_components` C).** The top 5 biopsies carry 62% (C1),
  80% (C5) and 86% (C6) of the summed loading, against 10–13% for C3/C4. Dropping them, the
  status association holds for C1 (Kruskal p 3e-16) and C5 (4e-17) but not for C6 (p 0.09).
  C6 is a few-biopsy component and is not interpreted further.
- **Scale caveat for the text.** C1's and C5's raw-scale η² is small because a few extreme
  biopsies dominate the variance. On loading ranks (the typical biopsy) status η² is 0.22
  (C1) and 0.24 (C5).

## Aim 2. The hormone programs resolve functional enteroendocrine states the annotation cannot encode

**Framing (literature).** Enteroendocrine (EE) hormone output is not a fixed cell-type
identity.
- A single EE cell passes through functional states with different hormone repertoires as it
  matures and migrates up the crypt–villus axis (Beumer, Gehart & Clevers 2020, *Endocr Rev*
  41:bnaa018; Gehart et al. 2019, *Cell* 176:1158).
- The switch is driven by a BMP gradient: GLP-1 (GCG) is high in the crypt, and secretin,
  PYY and NTS on the villus (Beumer et al. 2018, *Nat Cell Biol* 20:909). The human EE atlas
  is Beumer et al. 2020, *Cell* 181:1291.
- One-cell-one-hormone "letter" types are untenable: one major lineage co-expresses CCK,
  secretin, GIP, GLP-1, PYY and NTS (Egerod et al. 2012, *Endocrinology*; Fothergill & Furness
  2018, *Histochem Cell Biol*).
- IBDverse annotates a single "Enteroendocrine cells" type whose markers include GCG, PYY and
  NTS.
- So C3/C4-high cells are hormone *states* along this switch, not L-cell subtypes. Use
  "L-hormone+ EE" and "functional hormone states" in the text, never "L cells" as a type.

**2a. The hormone programs occupy distinct regions inside one annotated type (Fig 5E–J;
`figureS_ibd_hormone_states`).**
- **Method.** A PaCMAP of the EE, EC and tuft cells' RISE projections, zoomed to these
  three annotated types. `figureS_ibd_hormone_states` B shows where they sit on an all-cell map.
- E colours EE cells by whether they detect an L hormone (GCG, PYY or INSL5). This marker call
  is defined in `figureS_ibd_hormone_states` A, as Figure 2 defines lung epithelial cells by
  marker dot plot: 2,145 L-hormone+ and 913 other EE cells.
- F and G paint each cell's C3 and C4 sender weight, as Figure 2 paints weights on cells. The
  two programs peak in different EE regions.
- H and I show GCG and PYY expression (C4's and C3's top ligands) on the EE/EC part of the map.
- **J (sender weight, L-hormone+ vs other EE).** AUROC 0.83 (C3) and 0.73 (C4).
- **Robust to detection depth** (`figureS_ibd_hormone_robustness` A): 0.75–0.91 (C3) and
  0.69–0.77 (C4) across tertiles of genes detected, against 0.57 for depth alone.
- **Robust to status** (`figureS_ibd_hormone_robustness` B): 0.81–0.84 (C3) and 0.73–0.76
  (C4).
- **The programs are compact in eigenstate space** (`figureS_ibd_hormone_robustness` C).
  - Eigenstates needed for 80% of squared sender loading: C3 2, C4 1.
  - Eigenstates needed for 80% of the cells' mean squared projections: EE cells 6, EC cells
    3, secretory lineage 19.
  - So cccRISE uses a finer basis than the cells' overall variation.
- `figureS_ibd_hormone_states` C–F shows GCG, PYY, NTS and SST on the zoomed map.

**2b. The two states are crypt-like and villus-like (Fig 5K; `figureS_ibd_hormone_robustness`
D).**
- **Method.** Hormone states are EE cells in the top quarter of C3 (not C4) or of C4 (not C3)
  sender weight: 426 cells each. Genes are compared with Mann–Whitney tests, BH-corrected, and
  drawn as Figure 2's dot plot.
- **C3-high (villus-like) cells carry the BMP (villus-tip) targets.** Percent detected,
  C3-high vs C4-high: ID1 83% vs 18%, ID2 76 vs 42, ID3 75 vs 18, SMAD6 33 vs 3, SMAD7 49 vs
  13 (all q < 1e-4). They also carry PYY (91 vs 64%), NTS (94 vs 19%), SCT (88 vs 61%) and
  CCK.
- **C4-high (crypt-like) cells are GCG-dominant.** GCG is detected at a similar rate (77 vs
  79%), but its mean expression is 5.5× higher in C4-high cells (0.23 vs 0.04). They also carry the L-lineage
  transcription factors ARX, ISL1, PAX6 and GPBAR1 (`figureS_ibd_hormone_robustness` D).
- This matches the BMP-driven crypt-to-villus hormone switch (Beumer et al. 2018).
  - Don't argue from PYY or CCK: BMP did not raise PYY in human organoids, and CCK is not
    zonated there.
  - The robust readouts are ID1–3, NTS and SCT up, and GCG down.

**2c. Inflammation depletes the villus-like state (Fig 5L; `figureS_ibd_hormone_robustness`
E–F).**
- **L.** The villus-like (C3-high) share of each biopsy's EE cells falls with inflammation.
  Means are 17% healthy, 16% CD uninflamed and 9% CD inflamed; medians are 17%, 16% and 7%
  (Kruskal p 0.001; biopsies with at least 3 EE cells). The crypt-like share does not change
  (p 0.4).
- **What drops is carrier cells, not weight per carrier** (`figureS_ibd_hormone_robustness`
  E–F; per biopsy, medians healthy / CD uninflamed / CD inflamed):

  | Carrier | Cells per biopsy | Kruskal p | Weight per carrier | Kruskal p |
  |---|---|---|---|---|
  | EE senders | 7 / 10 / 5 | 6e-5 | 0.22 / 0.19 / 0.19 | 0.05 |
  | EC receivers (C3) | 2 / 3 / 1 | 0.001 | 0.35 / 0.30 / 0.37 | 0.1 |
  | Tuft receivers (C4) | 12 / 12 / 5 | 1e-8 | 0.24 / 0.23 / 0.29 | 0.005 (up in inflamed) |

  - This matches the covariate screen (1b), where secretory and stem-cell fractions track the
    uninflamed vs inflamed contrast as strongly as C3/C4 do.
- **Literature.** In small-bowel Crohn's, GLP-1 cells are increased and PYY cells unchanged
  with a downward trend (Moran et al. 2012, *J Crohns Colitis* 6:887). That is consistent
  with losing the villus-like PYY/NTS state while keeping the crypt-like GLP-1 state. The
  wider IBD EE literature is mixed.
- **Be honest that we can't tell** whether carriers are killed or crowded out of the biopsy by
  the infiltrate.

## Aim 3. The inflammatory programs are one coordinated, known regulatory circuit

**Framing (literature).**
- IL1R2 is a decoy receptor that captures IL-1 without signalling (Colotta et al. 1993,
  *Science* 261:472; Mantovani et al. 2019, *Immunity* 50:778).
- IL-1 signalling through NF-κB induces chemokines including CXCL8 (Weber, Wasiliew & Kracht
  2010, *Sci Signal* 3:cm1).
- CXC chemokines recruit neutrophils through CXCR1/2, and CC chemokines monocytes through CCR1
  (Griffith, Sokol & Luster 2014, *Annu Rev Immunol* 32:659).
- So C1 reads as IL-1 signalling into its decoy, a brake, and C5 as chemokine recruitment
  loaded against that decoy signal (Aim 1a). Keep this general; the text needn't cite every
  ligand–receptor edge.
- **IBD context.**
  - IL-1-driven, neutrophil-rich ulcer pathology marks therapy non-response (Friedrich et al.
    2021, *Nat Med*). Their circuit is stromal; this cohort excludes stroma.
  - Inflammatory mononuclear phagocytes characterize anti-TNF-resistant ileal CD (Martin et
    al. 2019, *Cell*, the GIMATS module).

**3a. The top pairs move together across biopsies (Fig 5M).** This is Figure 3D's test.
- **Method.** Each LR pair is scored per biopsy in the component's own sender/receiver context
  (`covariation.context_scores`, as Figure 3). Spearman correlations are taken across biopsies
  among the component's top 10 pairs. The baselines are 2,000 draws each of random,
  expression-matched and other status-associated pairs (`covariation.covariation_baselines`).
- **Median ρ, top 10 vs baselines:**

  | | Top 10 | Expression-matched | Random | Other associated |
  |---|---|---|---|---|
  | C1 | 0.46 | 0.32 | 0.08 | 0.05 |
  | C5 | 0.48 | 0.28 | 0.07 | 0.06 |

- The "other associated" baseline answers "is it just everything that rises with
  inflammation": it isn't.

**3b. Every top pair is higher in inflamed biopsies (`figureS_ibd_inflammatory_programs` A–B).**
This is Figure 3F's test.
- Each top pair's CD-inflamed minus healthy mean depth-normalized context score, against
  10,000 biopsy-label permutations.
- All 10 pairs of each component are higher in inflamed biopsies (BH q ≤ 7e-4).

**3c. The receivers express the ligands' downstream targets (Fig 5N).** This is Figure 2's
NicheNet test.
- Weighted target-score percentile against shuffled NicheNet priors, with receivers chosen
  among fine cell types (`cccrise-nichenet --receiver-celltype-column celltype`).
- C1 0.89 (q 0.006) and C5 0.78 (q 0.04) are significant; no other component is.
- C3's receiver test sits far below the null. Hormone GPCR targets are poorly covered by
  NicheNet's prior, so don't read it as evidence against C3.
- Sam's #233 note applies: the test validates receiver *cell types*.
- **Exploratory, not on the branch:** a label-free variant (top cells by receiver weight)
  keeps C1 significant at a 1% cut but not C5. That is consistent with C5's
  receptor-expressing cells (S100A8/9+ monocytes; CXCR1/2, IL1R2) sitting outside its
  receiver weight state. ALAD's significant components are unchanged under that variant.

**3d. What the high-weight cells express (`figureS_ibd_inflammatory_programs` C).**
- Figure 2's dot plot of the top 2% of myeloid cells by C1 and/or C5 sender weight.
- Both share an NF-κB / IL-1 activation state: IL1B, SOD2, NFKBIA, TNFAIP3, ICAM1, CCL3/4.
- C1-only cells are IFN-γ-activated (CXCL10, WARS, KYNU, IL4I1, SERPINB9).
- C5-only cells carry neutrophil-like genes (FCGR3B, HCAR2, CSF3R, IL1R2, S100A8, G0S2).
- Don't claim the program targets neutrophils.
  - **Exploratory, not on the branch:** within the label holding them ("Monocytes
    S100A8/9+"), C1/C5 weights do not separate neutrophil-like (FCGR3B+ CSF3R+) cells (AUROC
    about 0.5).
- **Robustness to tissue dissociation** is not tested on the branch. The status model
  adjusted for dissociation was dropped with the rank-OLS panel; `cccrise-ibd-programs`
  still writes the per-biopsy dissociation score.
  - **Exploratory, not on the branch:** the program genes persist in the least-dissociated
    third of myeloid cells, where IL1B, CXCL8 and CCL3/4 are 5–24× higher in high senders
    than in other low-dissociation myeloid cells.

## Caveats and what we leave out

- **C3 and C4 are each essentially one LR pair** (≈0.99 of their loading), so don't call them
  multi-pair programs. Their status signal is carrier abundance (1b, 2c).
- **C1 and C5 rest on few biopsies** (top 5 carry 62% and 80%). The association survives
  removing them (1b).
- **C6 is a few-biopsy component** (86% in the top 5) and is not interpreted.
- **Neutrophils are not recovered** by cccRISE as a population (3d).
- **No treatment data**, so no link to anti-TNF response.
- **Cell-level C1/C5 analysis is deferred to Figure 6:** weight vs contribution (#241),
  comparator recovery, and a resting macrophage state that shares the program's state without
  its ligands. The note is in the Figure 6 worktree.

## References (checked by abstract and record)

- Krzak et al. 2026, *Nat Genet*, doi:10.1038/s41588-026-02634-7 (IBDverse).
- Beumer et al. 2018, *Nat Cell Biol* 20:909–916, doi:10.1038/s41556-018-0143-y.
- Beumer et al. 2020, *Cell* 181:1291–1306, doi:10.1016/j.cell.2020.04.036.
- Beumer, Gehart & Clevers 2020, *Endocr Rev* 41:bnaa018, doi:10.1210/endrev/bnaa018.
- Gehart et al. 2019, *Cell* 176:1158–1173.
- Egerod et al. 2012, *Endocrinology* (PMC3440453).
- Fothergill & Furness 2018, *Histochem Cell Biol*, doi:10.1007/s00418-018-1746-x.
- Moran et al. 2012, *J Crohns Colitis* 6:887–894, doi:10.1016/j.crohns.2012.01.021.
- Colotta et al. 1993, *Science* 261:472–475, doi:10.1126/science.8332913.
- Mantovani, Dinarello, Molgora & Garlanda 2019, *Immunity* 50:778–795,
  doi:10.1016/j.immuni.2019.03.012.
- Weber, Wasiliew & Kracht 2010, *Sci Signal* 3:cm1 (PMID 20086235).
- Griffith, Sokol & Luster 2014, *Annu Rev Immunol* 32:659–702.
- Friedrich et al. 2021, *Nat Med*, doi:10.1038/s41591-021-01520-5.
- Martin et al. 2019, *Cell*, doi:10.1016/j.cell.2019.08.008.
- van den Brink et al. 2017, *Nat Methods* (dissociation-induced genes).
