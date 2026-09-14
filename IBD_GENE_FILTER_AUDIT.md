# IBDverse: mitochondrial and ribosomal gene audit

Audit of `/opt/IVCSC/full_IBD_cohort.ivcsc.h5ad` (Krzak et al. terminal-ileum
cohort: 1,185,861 cells × 33,538 genes, 343 samples), run 2026-09-14 to decide
what a mito/ribo filter should remove and where in the pipeline it belongs.

All per-gene count statistics below are cohort-wide totals read from `var`
(`total_counts`, `n_cells_by_counts`, `pct_dropout_by_counts`), which the source
file ships. Per-cell fractions come from a stratified subsample of 41,160 cells
(up to 120 per sample, all 343 samples, seed 0); per-cell mito% is also available
for every cell in `obs.pct_counts_gene_group__mito_transcript` and the two agree.

---

## 1. Naming conventions: the file already carries gene-group flags

`var` has four boolean columns from the Sanger pipeline, so the grouping does
**not** have to be inferred from symbol prefixes:

| `var` column | genes | share of all counts |
|---|---:|---:|
| `gene_group__mito_transcript` | 13 | 18.226% |
| `gene_group__mito_protein` | 13 | 18.226% |
| `gene_group__ribo_protein` | 164 | 9.670% |
| `gene_group__ribo_rna` | 0 | 0.000% |
| **any of the four** | **177** | **27.896%** |

`mito_transcript` and `mito_protein` select exactly the same 13 genes — the
`MT-` chromosome set. `ribo_rna` is empty because rRNA genes are absent from the
CellRanger GRCh38 reference this cohort was quantified against (`feature_types`
is `Gene Expression` for all 33,538 rows). The flags do not overlap each other.

`var_names` are Ensembl IDs; symbols live in `var["gene_symbols"]`. 24 symbols
are duplicated across two Ensembl IDs each, but none of them are mito or
ribosomal, so duplicate handling does not affect this filter.

### 1a. Where the shipped flags and the naming conventions disagree

Cross-tabulating the flags against symbol-prefix rules turns up four
discrepancies that matter:

**The flag misses three genuine cytosolic ribosomal proteins.** `RPL13`
(ENSG00000167526), `RPL18` (ENSG00000063177) and `RPSA` (ENSG00000168028) are
canonical, highly expressed ribosomal proteins that `gene_group__ribo_protein`
does not mark. These are not duplicate-symbol artifacts — each has a single
unique Ensembl ID. Together they are 0.433% of all counts, which is more than
`RPLP0` and more than the entire mito-ribosomal set. **Trusting the shipped flag
alone leaves them in.**

**`gene_group__ribo_protein` conflates cytosolic and mitochondrial ribosomes.**
Of its 164 genes, 85 are cytosolic (`RPS*`/`RPL*`/`RPLP*` plus `FAU` and
`UBA52`, whose products are cleaved to RPS30 and RPL40) and 79 are mitochondrial
ribosomal proteins (`MRPS*`/`MRPL*` plus `DAP3`, which is MRPS29). These are
different things: the cytosolic set is the technical/translational-state block
we want gone, whereas the MRPs are ordinary nuclear-encoded housekeeping genes
carrying only 0.282% of counts. Applying the flag wholesale removes both.

**`MTRNR2L*` is unflagged.** Eleven nuclear-encoded MT-RNR2 pseudogenes carry
0.239% of counts, concentrated in `MTRNR2L12` (0.177%) and `MTRNR2L8` (0.054%).
They are a well-known ambient/technical artifact and are not mitochondrially
encoded, so no `MT-` rule catches them.

**A naive `^RP[SL]` regex is wrong.** It sweeps up 21 unflagged genes that are
not ribosomal proteins at all — most importantly the ribosomal protein S6
*kinases* (`RPS6KA1`–`RPS6KA6`, `RPS6KB1`/`RPS6KB2`, `RPS6KC1`, `RPS6KL1`),
which are signaling genes, plus antisense transcripts (`RPL34-AS1`,
`RPS6KB2-AS1`) and readthrough loci (`RPL36A-HNRNPH2`, `RPS10-NUDT3`,
`RPL17-C18orf32`). The shipped flag correctly excludes all of these.

**Conclusion on conventions:** use the shipped flags as the base and add a short
curated list; do not use prefix regexes as the primary rule, and do not use the
flags unmodified.

---

## 2. Composition and expression

### 2a. These genes dominate the count distribution

The 13 `MT-` genes alone are 18.2% of all counts in the cohort. Nine of the ten
most-expressed genes in the dataset are either mitochondrial or immunoglobulin:

| rank | gene | group | % of all counts | cumulative |
|---:|---|---|---:|---:|
| 1 | IGKC | — | 3.563 | 3.563 |
| 2 | MT-CO3 | MT | 3.242 | 6.805 |
| 3 | MT-CO1 | MT | 3.209 | 10.014 |
| 4 | MT-CO2 | MT | 2.673 | 12.687 |
| 5 | MT-ATP6 | MT | 2.306 | 14.993 |
| 6 | MT-ND3 | MT | 1.838 | 16.831 |
| 7 | MALAT1 | — | 1.683 | 18.515 |
| 8 | MT-CYB | MT | 1.671 | 20.185 |
| 9 | MT-ND4 | MT | 1.323 | 21.509 |
| 10 | IGHA1 | — | 0.918 | 22.427 |

The top 30 genes are 30.7% of all counts, and 19.2 of those 30.7 points are
mitochondrial or ribosomal. Put the other way: **0.53% of the genes carry 27.9%
of the counts.**

They are also the least sparse genes in the dataset. Median dropout across all
33,538 genes is 99.49%; across the 177 flagged genes it is 65.41%, and 158 of
the 177 have dropout below 90%. Every one of them clears the loader's
`gene_threshold` and enters the working gene set.

### 2b. Per-cell burden

Mean fraction of a cell's counts, over the 41,160-cell stratified subsample:

| group | genes | mean | median | 75th | 95th | max |
|---|---:|---:|---:|---:|---:|---:|
| `MT-` | 13 | 13.74% | 9.26% | 19.32% | 41.80% | 79.92% |
| cytosolic ribosomal (flagged) | 85 | 11.77% | 9.81% | 17.56% | 31.78% | 79.45% |
| cytosolic ribosomal (missed by flag) | 3 | 0.53% | 0.43% | 0.79% | 1.49% | 6.81% |
| `MTRNR2L*` | 11 | 0.32% | 0.14% | 0.33% | 1.21% | 25.38% |
| mito ribosomal (`MRP*`, `DAP3`) | 79 | 0.27% | 0.25% | 0.36% | 0.56% | 2.24% |
| **all of the above** | **191** | **26.63%** | **25.53%** | — | **53.37%** | **97.20%** |

The 95th-percentile cell spends over half its library on these genes, and the
worst cells spend essentially all of it.

### 2c. The burden is confounded with disease status

Per-sample means (each sample's mean over its cells, then summarized across
samples):

| group | n samples | `MT-` | cytosolic ribo | combined |
|---|---:|---:|---:|---:|
| Healthy | 232 | 14.71% | 11.98% | 27.85% |
| CD uninflamed | 47 | 13.70% | 11.41% | 26.18% |
| CD inflamed | 64 | 10.26% | 11.26% | 22.54% |

Healthy samples carry **4.5 percentage points more** mitochondrial signal than
CD-inflamed samples, and 5.3 points more combined. This runs along exactly the
axis the analysis is trying to discriminate, and it lives in a channel that is
technical (dissociation stress, cell viability, ambient RNA) rather than
biological. The full-cohort per-cell mito% from `obs` reproduces this
independently: Healthy 14.73% vs CD 11.83%.

Ribosomal content, by contrast, is **not** materially confounded with disease
(11.26–11.98% across all three groups). The disease confound is a mitochondrial
effect specifically.

### 2d. The burden is strongly confounded with cell type

| cell type | `MT-` | cytosolic ribo | combined |
|---|---:|---:|---:|
| Stem cell | 23.11% | 13.15% | 37.69% |
| B cell | 7.00% | 22.76% | 31.36% |
| Enterocyte | 21.75% | 7.92% | 30.63% |
| T cell | 8.23% | 15.76% | 25.33% |
| Secretory | 18.03% | 3.99% | 22.84% |
| Myeloid | 6.78% | 14.22% | 21.99% |
| Mesenchymal | 7.58% | 8.38% | 16.87% |
| Plasma B cell | 6.06% | 7.73% | 14.49% |

A 3.8-fold range in mito content and a 5.7-fold range in ribosomal content
across lineages — **and the two run in opposite directions.** Epithelial
lineages (Stem, Enterocyte, Secretory) are mito-high and ribo-low; lymphoid and
myeloid lineages are the reverse. Because the communication tensor is indexed by
sender × receiver cell type, this is a systematic technical gradient laid
directly across two of the tensor's four modes.

### 2e. Per-sample spread

Across the 343 samples, per-sample mean fractions range:

| group | min | 5th | median | 95th | max |
|---|---:|---:|---:|---:|---:|
| `MT-` | 3.96% | 6.56% | 13.51% | 21.01% | 35.94% |
| cytosolic ribo | 0.20% | 6.53% | 10.87% | 20.27% | 28.00% |
| combined | 14.18% | 19.07% | 27.00% | 33.20% | 39.36% |

A 9-fold range in mito fraction across samples, and samples are the condition
mode of the RISE decomposition.

---

## 3. Why this changes the results, and where the filter has to go

**No mitochondrial or ribosomal gene is a ligand or receptor.** Checked against
all 990 distinct ligand/receptor subunit symbols in
`Human-2020-Jin-LR-pairs.parquet`: the intersection with the 177 flagged genes is
empty, and so is the intersection with the proposed additions. Filtering them
therefore removes **zero** communication signal directly.

Their entire effect is indirect, and it runs through normalization. The loader
(`vsparse.load_and_normalize`, reproducing `prepare_dataset`) does, in order:

1. drop cells with total counts ≤ `min_cell_counts`;
2. drop genes with raw total ≤ `gene_threshold × n_cells`;
3. divide each cell by its total counts, rescaled to the median cell depth;
4. divide each gene by its column sum;
5. `log10(1000x + 1)`.

Step 3 is the problem. The per-cell denominator currently includes the 27.9% of
counts these genes contribute, and that 27.9% varies from 14% to 39% across
samples and from 14% to 38% across cell types. **Every other gene's normalized
value is therefore scaled by a sample- and cell-type-specific factor that is
technical in origin.** Step 4's median-depth rescaling compounds it, since the
median is taken over the cells that survive.

This has a direct consequence for how the filter must be implemented:

> **The exclusion must be applied before normalization, not by subsetting
> columns afterwards.** Dropping the genes post-hoc leaves the per-cell depth and
> the median rescaling computed over a gene set that still includes them, which
> preserves the confound this filter exists to remove.

This is the same failure mode recorded for condition subsetting: normalization
scope must match fit scope.

Steps 1, 2, 4 and 5 are unaffected by the order — the gene threshold is a
per-gene test against raw totals, and the per-gene column sums are per-gene — so
applying the exclusion at step 2 gives exactly the right answer.

---

## 4. Recommended exclusion set

| # | set | genes | % of counts |
|---|---|---:|---:|
| A | `gene_group__mito_transcript` (the `MT-` genes) | 13 | 18.226% |
| B | `gene_group__ribo_protein` minus `MRP*`/`DAP3` (cytosolic) | 85 | 9.388% |
| C | `RPL13`, `RPL18`, `RPSA` — cytosolic RPs the flag misses | 3 | 0.433% |
| D | `MTRNR2L*` nuclear MT-RNR2 pseudogenes | 11 | 0.239% |
| | **recommended total (A+B+C+D)** | **112** | **28.285%** |
| E | `MRPS*`/`MRPL*`/`DAP3` — mito ribosomal proteins | 79 | 0.282% |
| | if E is also excluded | 191 | 28.568% |
| | shipped flags used unmodified, for comparison | 177 | 27.896% |

33,426 of 33,538 genes survive the recommended filter. After the loader's own
`gene_threshold=0.001` is also applied, the working gene set goes from 23,729
genes to **23,622** — only 107 fewer, because five of the 112 would have failed
the count threshold anyway. Verified end to end on the full cohort: 116 s,
80.5 GB peak RSS, 3,550,747,821 nonzeros, all 343 conditions intact, no `MT-`,
`MTRNR2L*` or cytosolic ribosomal gene remaining, and all 79 `MRP*` retained.

**Two judgement calls are embedded here, both cheap to reverse:**

- **`MRP*` are kept (E excluded from the filter).** They are nuclear-encoded
  housekeeping genes, not a technical artifact, and carry 0.28% of counts. The
  shipped `ribo_protein` flag disagrees. Excluding them too changes the removed
  fraction by 0.28 percentage points, so the choice is nearly free either way —
  it is a question of what the filter is *for*, not of magnitude.
- **`MTRNR2L*` are removed (D included).** They are a recognized ambient-RNA
  artifact and no `MT-` rule catches them, but they are nuclear genes and a
  strict "mitochondrially encoded" reading would keep them.

Both are single-constant edits in the implementation.

---

## 5. What this does not tell us

The audit measures composition and confounding. It does **not** measure the
effect on the decomposition itself — whether filtering moves the RISE rank, the
CP rank, or the recovered components. That requires re-running the sweeps on the
filtered data, which is the reason this work was sequenced before the IBDverse
rank re-runs rather than after.

The ALAD cohort has not been audited. Its `var` does not carry the Sanger
`gene_group__*` flags, so a symbol-based rule would have to be built and
validated separately before the same filter could be applied there. The
committed ALAD rank results are unaffected by this change.
