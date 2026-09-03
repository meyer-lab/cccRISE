# CPD rank selection: ALAD vs IBDverse, on an optimized sweep

**Scope note up front:** this documents two things. First, the changes that make
the interaction-tensor CP rank sweep affordable at cohort scale — CANDELINC
compression and GPU CP-ALS, both on by default and both opt-out. Second, the
result of running that sweep on both datasets. The **selection rule is
deliberately not implemented in code**: the sweep reports the metrics and the
rank is read off the plot, the same way the BiCV RISE sweep is read. RISE rank 25
is pre-selected and every CP number below is conditional on it.

The companion HTML report (`scratch/cpd_rank_report.html`) presents the same
evidence with interactive figures. All numbers are measured on this repository's
cached interaction tensors; nothing is extrapolated.

## The answer

| | ALAD | IBDverse |
|---|---|---|
| **Selected CP rank** | **4** | **2** |
| CORCONDIA at that rank | 91.30% | 99.96% |
| at rank + 1 | −79.98% | −857.1% |
| at rank + 2 | −3176% | −934.0% |
| Any threshold in… | (−80.0, 91.30) picks it | (−857, 99.96) picks it |
| Fit R2X at that rank | 0.462 | 0.425 |
| First canceling pair | rank 6 | rank 5 |
| Tensor | 47 × 25 × 25 × 513 (15.1M) | 343 × 25 × 25 × 1372 (294M) |
| RISE R2X | 0.0404 | 0.0335 |

Both datasets show the same qualitative signature: core consistency sits near
100%, then in a single rank step **changes sign** — the best-fit core stops being
near-superdiagonal, meaning components have begun trading off against each other
rather than each explaining independent structure — and keeps falling by orders
of magnitude after that. They just flip at different ranks. Because the step is a
sign change rather than a gradual decay, neither answer depends on the 50%
threshold: any bound between the two bracketing values picks the same rank.

The collinearity screen corroborates without coinciding. It is looser on both,
first flagging a canceling pair at ALAD rank 6 and IBD rank 5 — in both cases
above the selected rank, so the two criteria bracket rather than contradict.

**ALAD rank 4 survives the new pipeline unchanged**, reproducing the earlier
audit's numbers to the decimal (100.0 / 97.5 / 91.5 / 91.3 / −80.0 / −3174.5
there; 100.0 / 97.53 / 91.53 / 91.30 / −79.98 / −3176.18 here).

## What changed

| Change | Where |
|---|---|
| CANDELINC compression + R2X restatement | `rank_selection.candelinc_compress`, `candelinc_r2x` |
| GPU CP-ALS with a CPU fallback | `ccc_rise.cp_decomposition(device=...)` |
| `--no-compress`, `--oversample`, `--device` | `runners/run_cpd_rank.py` |
| CP fitting controls split from RISE's (`--cp-n-iter-max`, `--cp-tol`) | `runners/run_cpd_rank.py` |
| Host SVD init so ALS stays on the GPU when cuSOLVER refuses | `ccc_rise._host_svd_init` |
| IBD source path moved to `/opt/IVCSC` | `import_data.IBD_FULL_COHORT_IVCSC` |
| IBD Ensembl→symbol re-indexing | `import_data.use_gene_symbols` |
| Fail-loud guard when no LR pair matches | `ccc_rise.calc_communication_score` |

The metric functions themselves (`cp_core_consistency`,
`cpd_component_collinearity`) are **unchanged** — they did not need to change,
which is the point of the next section.

The CP/RISE split matters more than it looks: CORCONDIA collapses on an
under-converged fit, so CP wants `n_iter_max=4000, tol=1e-12`, which would be
absurd for the RISE fit the same flag used to control. On the compressed core
that cap costs almost nothing.

## Why compressing is safe

Both metrics are invariant to the compression. The bases are orthonormal, so
within-mode cosines between decompressed components equal those between core
components; and `pinv(U F) = pinv(F) Uᵀ`, so the CORCONDIA core is identical
whether projected out of the original tensor or the compressed one. Measured on
the real IBD tensor, the two agree to **2×10⁻¹⁵ relative**. R2X is the one
quantity that does change, and `candelinc_r2x` adds back the discarded sum of
squares so the CSV still reports R2X against the original tensor.

Fitting *inside* the subspace is a genuine restriction, so that was checked
separately rather than assumed:

| Check | Result |
|---|---|
| Metric off core vs off full tensor, same factors (IBD) | identical to 2×10⁻¹⁵ relative |
| **ALAD, uncompressed at all 25 ranks** | CORCONDIA to 0.04%, R2X to 4.6×10⁻⁵ through rank 4 |
| **IBD, uncompressed at ranks 1–8** | CORCONDIA to 0.002%, R2X to 8.0×10⁻⁵ through rank 2 |
| Cliff location vs retained variance (ALAD 88.0→99.2%) | unchanged at rank 4 |
| Cliff location vs retained variance (IBD 86.5→99.2%) | unchanged at rank 2 |
| Same IBD comparison under *random* init | disagrees by up to 0.08 R2X — **in both directions** |

Both datasets were fit twice at each rank — once inside the CANDELINC core, once
against the full tensor — from the same deterministic SVD initialisation with the
same 4000-iteration budget, so the only difference is the subspace constraint.

**Coverage differs by dataset, deliberately.** ALAD is cheap enough to run
uncompressed at all 25 ranks (2 minutes total). IBD is not: even with ALS on the
GPU its uncompressed fits cost 3–8 minutes each, so it was run through **rank 8**
— four ranks past its cliff, far enough to establish the pattern — and stopped
there rather than spending hours confirming behaviour in a region the selection
never reaches.

Both cliffs reproduce independently on the uncompressed fits:

| | rank | compressed | uncompressed | rel. diff |
|---|---|---|---|---|
| ALAD, at the cliff | 4 | 91.3009 | 91.3364 | 0.04% |
| ALAD, one past | 5 | −79.982 | −78.018 | 2.5% |
| IBD, at the cliff | 2 | 99.9559 | 99.9581 | 0.002% |
| IBD, one past | 3 | −857.13 | −853.47 | 0.43% |

The collinearity screen returns an identical mode count at 20/25 ALAD ranks and
8/8 IBD ranks.

Past each cliff the two drift apart — ALAD reaching 8% relative by rank 6, IBD 7%
by rank 8 — while R2X never follows, tracking to 1.8×10⁻² across every fit. That
is the degeneracy signature, not compression error: past the true rank many
solutions fit equally well, so the subspace constraint picks a different one and
CORCONDIA stops being a stable quantity.

The random-init row is the informative one. A subspace restriction can only ever
cost variance, never add it, so a two-sided gap is the seed lottery on a rugged
objective, not the compression. That is why the sweep initialises
deterministically (`init="svd"`).

The identifiability boundary is asserted at that scope in
`test_runner_compressed_and_uncompressed_agree`: the two paths must agree where
the model is identifiable, and must both see it degrade past that, but are not
required to agree on a number in the degenerate region.

## Cost — and when it matters

**The optimization only pays at cohort scale.** On ALAD it is worth about 4×; the
tensor is small enough that fixed overhead dominates. On IBD, 20× larger, the
same code is worth four orders of magnitude per fit.

| | ALAD | IBDverse |
|---|---|---|
| Tensor entries | 15,069,375 | 294,122,500 |
| Core | 30×25×25×30 | 30×25×25×30 |
| Compression | 27× | 523× |
| Retained | 88.0% | 86.5% |
| Sweep, ranks 1–25 | **32 s** | **68 s** |
| Same sweep uncompressed | 126 s (25 ranks) | 40 min (ranks 1–8) |
| Speed-up | 4× | 1,317× (528–4,181× per fit, median 1,257×) |

The two effects are separable, and worth separating carefully. Measured on the
uncompressed IBD tensor at matched random init and iteration budget, **the GPU
alone is worth 6.1× at rank 3 and 15.9× at rank 8** (identical R2X to four
decimals). Compression supplies the rest.

An earlier draft of this report quoted 5,165–11,557× for the compression on IBD.
That figure compared against a *CPU-only* uncompressed fit and so silently
credited the compression with the GPU's contribution too. Once the host-init fix
below lets the uncompressed baseline use the GPU, the honest number for
compression alone is the 1,317× in the table.

One trap found the hard way: **cuSOLVER cannot SVD-initialise the uncompressed
IBD tensor** — its 343 × 858,125 mode-0 unfolding fails with
`CUSOLVER_STATUS_INVALID_VALUE`.

Only the *initialisation* needs an SVD, though; ALS is ordinary matmuls.
`cp_decomposition` now builds the init with NumPy, copies it to the device and
retries, keeping every iteration on the GPU — about 170 s of host init plus
~0.1 s/iteration, so an uncompressed cohort-scale fit costs minutes rather than
hours. That is what made the IBD comparison above possible at all. A blanket CPU
fallback remains as a last resort for out-of-memory. The `gpu` extra pins
`cupy-cuda12x[ctk]`, since without the CUDA toolkit headers the retry cannot
JIT-compile and drops to the CPU anyway.

## Both tensors are spike-dominated

| Largest entries | 1 | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|---|
| ALAD share of sum of squares | 5.09% | 56.34% | 80.77% | 94.26% | 99.06% |
| IBD share of sum of squares | 10.00% | 50.85% | 78.39% | 94.27% | 98.98% |

Of ALAD's 15.1M entries the largest 100 hold over half the sum of squares; IBD is
the same story at 20× the size. `expression_product` multiplies two RISE
eigen-state expression values, so it is heavy-tailed by construction, and every
criterion in use is least-squares based — so all of them are effectively scored
on a few hundred entries. The concentration is the communication signal, not a
defect to transform away, but it is what makes the objective rugged and
deterministic fitting necessary.

## Caveat that applies to both

The underlying RISE fits explain **4.04%** (ALAD) and **3.35%** (IBD) of
expression variance, so the eigen-states these interaction tensors are built from
are themselves weakly determined. The CP rank question is well posed and the
cliffs are sharp; how much biology sits under them is a separate question neither
sweep answers.

## Reproducing

```
uv sync --extra gpu
uv run ccc-rise-cpdrank --dataset alad --rise-rank 25 --ranks 1:26
uv run ccc-rise-cpdrank --dataset ibd  --rise-rank 25 --ranks 1:26
uv run fbuild CPDrank
```

Compression and GPU are the defaults; `--no-compress` and `--device cpu` recover
the original behaviour. The sweep takes under two minutes on either dataset — the
single RISE fit that produces the interaction tensor still dominates the runner.

## What was not done, and why

- **A hardcoded selection rule.** Deliberate: the rank is eyeballed off the plot,
  as with BiCV. The metrics the earlier audit found unusable (worst-pair Tucker
  congruence, restart agreement) are also not carried over.
- **Interaction-tensor / RISE-fit caching in the runner.** The cached artifacts in
  `cellcommunicationpf2/data/` were used for the scratchpad analysis, but the
  runner still refits RISE each time. Worth adding separately.
- **A provenanced `cpd_rank_ibd.csv`.** The IBD numbers here come from the
  scratchpad against the cached tensor, so no sidecar was produced. Running the
  command above produces both.
- **IBD uncompressed above rank 8.** Stopped there on purpose: the remaining 17
  ranks would cost hours and only characterise the degenerate region, which
  ALAD's full 25-rank comparison already shows. The partial run is checkpointed
  in `scratch/full_uncompressed_ibd.csv` and resumes from rank 9 if wanted.
