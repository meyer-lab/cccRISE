# Runners

Scripts that take data plus user inputs, run a CCC-RISE-related computation,
and **save the result** for use by another runner or a figure module. Runners
are the producers; `figures/` is the consumer. Nothing in
`cellcommunicationpf2/` imports a runner at library level, so a runner can be
as slow and as opinionated as it needs to be.

Every runner writes two files with the same stem:

| file | contents |
| --- | --- |
| `<name>.nc` / `<name>.csv` | the result itself |
| `<name>.provenance.yaml` | provenance sidecar in the schema [issue #164](https://github.com/meyer-lab/cellcommunication-Pf2/issues/164) documents: source file + sha256, loader + args, condition key, ranks and their evidence, fitting parameters, `resulting_dims`, R2X values, `git_sha`, `package_versions`, `generated` |

Results land in `cellcommunicationpf2/data/` by default, under the exact
filenames the figure modules already look for, so re-running a runner
refreshes the artifacts in place.

## Datasets

Pick a dataset with one word. Everything else — loader, raw source file,
condition column, output filenames — comes from the registry in
[`datasets.py`](datasets.py).

| key | dataset | condition column | factors file |
| --- | --- | --- | --- |
| `alad` | ALAD lung transplant BAL | `dsco_id` | `bal_alad_factors.nc` |
| `ibd` | IBDverse terminal ileum | `sanger_sample_id` | `ibd_ti_factors.nc` |

Add a dataset by appending a `DatasetSpec` to `DATASETS`; nothing else needs
to change.

## The three runners

### 1. RISE rank selection — `run_rise_rank.py`

Bi-cross-validation sweep over RISE ranks (`rank_selection.run_bicv`).

```bash
uv run ccc-rise-riserank --dataset alad --ranks 5:51:5 --n-folds 5 \
    --selected-rank 25 --notes "elbow of the held-out block R2X curve"
```

Writes `data/rise_rank_bicv_alad.csv` (read by `figureRISErank`) and
`data/rise_rank_bicv_alad.yaml`.

### 2. CPD rank selection — `run_cpd_rank.py`

Fits RISE at an already-selected rank, builds the interaction tensor, then
sweeps CP ranks with entry-masked imputation CV
(`rank_selection.cp_masked_reconstruction_sweep`) and, with `--fms-runs`, the
FMS-elbow criterion (`tensor.run_fms_r2x_analysis`) — the two criteria issue
#164 asks for. CP rank selection is *conditional* on the RISE rank, which is
why `--rise-rank` is required and appears in the output filename.

```bash
uv run ccc-rise-cpdrank --dataset alad --rise-rank 25 --ranks 1:26 --n-masks 5 \
    --fms-runs 3 \
    --selected-rank 14 --notes "plateau of held-out entry R2X, FMS elbow at 14"
```

Writes `data/cpd_rank_imputation_alad_rise25.csv` (read by `figureCPDrank`),
`..._fms.csv` when `--fms-runs > 0`, and the matching `.provenance.yaml`. The
interaction tensor is cached in the system temp directory keyed by everything
that determines it, so extending a rank range skips the RISE refit; entries
older than a day are pruned.

Ranks above the tensor's smallest mode are dropped from the sweep and recorded
in `cp_ranks_skipped_over_max_mode` — see the rank cap below.

### 3. The pipeline — `run_ccc_rise.py`

The end-to-end decomposition at the ranks the two sweeps justified.

```bash
uv run ccc-rise-run --dataset alad --rise-rank 25 --cp-rank 14
```

Writes `data/bal_alad_factors.nc` and `data/bal_alad_factors.provenance.yaml`.
The `.nc` is read by the figure modules via `figures.common.get_alad_data` /
`factors.load_factors_to_anndata`.

## The CP rank cap

`ccc_rise.validate_cp_rank` rejects `cp_rank > min(tensor.shape)`. The
communication tensor's modes are `(n_conditions, rise_rank, rise_rank,
n_lr_pairs)`, so in practice the binding constraint is
`min(n_conditions, rise_rank)`.

Past that point tensorly's `initialize_cp` with `init="svd"` falls back to a
full SVD of a mode unfolding — for ALAD's published `rise 15 / cp 18` that is a
361,665 × 361,665 array, a 975 GiB allocation — and pads the excess components
with uniform random columns, so they start from noise rather than from the
data. The pipeline runner checks this in seconds, before the RISE fit, so a bad
rank pair fails immediately rather than an hour in.

## Recording *why* a rank was chosen

The pipeline sidecar has a `rank_selection` block with one entry per stage:

```yaml
rank_selection:
  rise:
    rank: 25
    method: bicv
    sweep_csv: cellcommunicationpf2/data/rise_rank_bicv_alad.csv
    fms_csv: null
    sweep_yaml: cellcommunicationpf2/data/rise_rank_bicv_alad.provenance.yaml
    sweep_generated: '2026-08-31T18:04:11+00:00'
    sweep_git_commit: 7a8e97a4f3a51cb1c01a1ce6311cd8b6139cef88
    sweep_selected_rank: 25      # what --selected-rank recorded on the sweep
    notes: elbow of the held-out block R2X curve
  cpd:
    rank: 14
    method: masked_entry_imputation_cv + fms_elbow
    ...
```

The block is auto-populated from the
sweep sidecars when they sit at their canonical paths; point elsewhere with
`--rise-rank-selection` / `--cpd-rank-selection`, and state the rationale with
`--rise-rank-notes` / `--cpd-rank-notes`. Every cited CSV is also listed flat
in the top-level `rank_selection_evidence`, per the issue's schema.

## Source-data provenance

Every sidecar records `source_file` and `source_sha256`. This is the field that
would have caught the ALAD drift in issue #164: the stored factors claimed 445
LR pairs and 15,899 genes, but the same loader against today's source yields
513 and 16,675 — a change nothing in the repository recorded. Hashing a
multi-GB `.h5ad` takes a minute or two, negligible beside the fit; pass
`--skip-source-checksum` if it genuinely matters, and the field says `skipped`
rather than going silently missing.

`notes` fields are the only ones meant to be hand-edited, and a re-run
overwrites them — keep the reasoning in the PR too.

## Typical sequence

```bash
uv run ccc-rise-riserank --dataset alad --ranks 5:51:5 --selected-rank 25 \
    --notes "elbow of the held-out block R2X curve"
uv run ccc-rise-cpdrank  --dataset alad --rise-rank 25 --ranks 1:26 --fms-runs 3 \
    --selected-rank 14 --notes "plateau of held-out entry R2X, FMS elbow at 14"
uv run ccc-rise-run      --dataset alad --rise-rank 25 --cp-rank 14
make output/figureRISErank.svg output/figureCPDrank.svg
```
