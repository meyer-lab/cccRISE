# R-vs-Python checks for the vendored comparators

`cccrise/comparators/{scriabin,niches,_wgcna}.py` re-implement Scriabin's
interaction-program workflow and NICHES' cell-to-cell / system modes. These
scripts run the pinned upstream R code on the same inputs, with Python's random
draws fed into R through shims, and compare every stage.

Upstream pins: scriabin `bce695b`, NICHES `5d932b8`, WGCNA 1.74,
dynamicTreeCut 1.63-1, Seurat 5.6.0 (R 4.5.2).

```sh
# one-off: isolated R library (git-excluded) and upstream sources
Rscript rcheck/install.R                     # Seurat, WGCNA, qlcMatrix, ... into rcheck/lib
Rscript rcheck/install_faithful.R            # ALRA, SeuratWrappers, LISI, rsvd 1.0.5
git clone https://github.com/BlishLab/scriabin rcheck/upstream/scriabin && git -C rcheck/upstream/scriabin checkout bce695b
git clone https://github.com/msraredon/NICHES rcheck/upstream/niches && git -C rcheck/upstream/niches checkout 5d932b8
R CMD INSTALL -l rcheck/lib rcheck/scriabin_stub   # exposes scriabin::mapvalues / LoadLR only

# stage checks, per dataset (two samples each; cccRISE's expression / genes / LR pairs)
for d in alad ibd; do
  D=rcheck/out/check_$d
  uv run python rcheck/export_inputs.py $D --dataset $d
  for stage in programs significance scores; do      # programs also writes random_sets.json
    Rscript rcheck/run_upstream.R $D $stage
    uv run python rcheck/compare.py $D $stage
  done
done

# WGCNA primitives at upstream and non-default module-discovery settings
uv run python rcheck/wgcna_compare.py rcheck/out/wgcna write rcheck/out/check_alad rcheck/out/check_ibd
Rscript rcheck/wgcna_check.R rcheck/out/wgcna
uv run python rcheck/wgcna_compare.py rcheck/out/wgcna compare

uv run pytest -n 0 cccrise/tests/test_comparators_rcheck.py   # reruns all comparisons
```

Result (2026-11-02, `sc-comparator-concordance`): ALAD 95/95, IBD 137/137,
WGCNA 184/184 (4 similarities x 9 settings).

`rcheck/patched_ip.R` exposes the two module-discovery settings upstream
hard-codes (soft-power cap 3; tree cut at the 0.5 height quantile, its
`tree.cut.quantile` argument being ignored) for the parameter sweep.
`rcheck/shared/` exports a whole dataset for the R-shared arm (upstream R on
cccRISE's inputs, run per sample by `rcheck/faithful/ip_runs.R`).

`rcheck/scriabin_stub` is a two-file package holding upstream's
`mapvalues` verbatim (`InteractionPrograms` calls `scriabin::mapvalues`);
installing the real package would pull in CelliD, nichenetr and others that
the checked stages never call.

## Seed-to-seed stability (RNG audit)

The checks above feed Python's draws into R, so they cannot show whether the
port's *randomness* behaves like upstream's. `seed_stability.{R,py}` run
`InteractionPrograms` five times per sample with each language's own RNG and
compare module overlap between runs. Upstream calls `set.seed(1111)` inside
`InteractionPrograms`, so a caller's `set.seed` has no effect; the seed is
passed as `seed`. Cell types under `min.cell = 3` cells are dropped first
(upstream errors on them).

```sh
Rscript rcheck/seed_stability.R rcheck/out/check rcheck/out/seeds
uv run python rcheck/seed_stability.py rcheck/out/seeds
```

Result (2026-10-07; mean best-Jaccard between runs): ALAD1-HS1-E1 R 0.21,
Python 0.21, R vs Python 0.24; ALAD1-HS40-E1 R 0.25, Python 0.19, R vs Python
0.23. Per-sample modules depend on the 300-cell subsamples to this degree in
upstream itself.

## Leaving out one cell (no randomness)

`cell_drop.{R,py}` take the RNG out entirely: upstream `InteractionPrograms`
on each sample's full CCIM (`iterate.threshold` above the sample size, so no
subsampling), on all cells and with one cell left out, against the port on
the same cells.

```sh
uv run python rcheck/export_inputs.py rcheck/out/drop ALAD1-HS59-E1 ALAD1-HS24-E1 ALAD1-HS18-E1
Rscript rcheck/cell_drop.R rcheck/out/drop rcheck/out/drop/r 0 37 74
uv run python rcheck/cell_drop.py rcheck/out/drop rcheck/out/drop/r 0 37 74
```

Result (2026-10-08): R and Python modules identical in all 12 runs. Leaving
out one of 304-522 cells changes upstream's own modules: agreement with the
all-cell modules 0.73-0.99 (median 0.88). The network is nearly flat: soft
power falls back to 1 (the scale-free fit passes 0.6 only at power >= 6,
above upstream's cap of 3; 43 of 47 ALAD samples), so adjacency is
`0.5 + 0.5 * cor` with LR-pair correlations across cell pairs near 0, and
the dendrogram's merge heights span about 0.014 with a median gap of 1-2e-5
between consecutive merges. One cell moves similarity entries by 0.005-0.03,
far more than those gaps, so the tree reorders.

The 300-cell subsamples add most of the seed-to-seed change on top
(scratch experiments, not scripted): between seeds the averaged similarities
correlate at only r = 0.8-0.86 for HS1 / HS37, and module agreement is
0.2-0.3. Averaging 128 subsamples instead of upstream's
`round(sqrt(N / 300))` raises agreement for mid-sized samples (HS18 0.40 ->
0.74, HS40 0.31 -> 0.72) but not the largest (HS37 0.19 -> 0.18). Forcing
soft power 3 or 6 raises agreement by at most 0.23 (0.20-0.38 -> 0.25-0.51).

## Faithful Scriabin (authors' setup, no port)

`rcheck/faithful/` runs upstream Scriabin exactly as published: raw ALAD
counts, SCTransform regressing % mt / % ribo, ALRA, OmniPath (curation >= 7),
upstream defaults. On top of it, the seed / cell-loss / one-cell / UMI-depth
analyses, the paper's LISI noise test (Fig. 2d-e) and its Extended Data Fig.
8f quantities, plus a comparison with the port. Extra R packages:
`Rscript rcheck/install_faithful.R`; the stub now also carries `LoadLR` and
`lr_resources.rds`. Pipeline: `bash rcheck/faithful/run_all.sh`; tables:
`uv run python rcheck/faithful/compare.py`. Write-up:
`SCRIABIN_FAITHFUL_REPORT.md`.
