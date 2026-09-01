# Runners

Scripts that take data plus user inputs, run a CCC-RISE-related computation,
and **save the result** for a later runner or a figure module to read. Runners
are the producers; `figures/` is the consumer, and figures only plot what a
runner already wrote. Nothing in the library imports a runner, so a runner can
be as slow and as opinionated as it needs to be.

Each runner writes its result plus a sidecar:

| file | contents |
| --- | --- |
| `<name>.nc` / `<name>.csv` | the result itself |
| `<name>.provenance.yaml` | source file + sha256, loader + args, condition key, ranks and the sweeps that justify them, fitting parameters, `resulting_dims`, R2X values, `git_sha`, `package_versions`, `generated` |

Results land in `cellcommunicationpf2/data/` under the filenames the figure
modules read, so re-running a runner refreshes the artifacts in place.

## Datasets

Pick a dataset with one word. The loader, source file, condition column and
output filenames come from the registry in [`datasets.py`](datasets.py).

| key | dataset | condition column | factors file |
| --- | --- | --- | --- |
| `alad` | ALAD lung transplant BAL | `dsco_id` | `bal_alad_factors.nc` |
| `ibd` | IBDverse terminal ileum | `sanger_sample_id` | `ibd_ti_factors.nc` |

Output filenames carry only the dataset. The method, the ranks, and everything
else that distinguishes one run from another live in the sidecar, so a re-run
replaces the artifact rather than accumulating variants beside it.

Add a dataset by appending a `DatasetSpec` to `DATASETS`; nothing else changes.

## The three runners

### 1. RISE rank selection — `run_rise_rank.py`

Bi-cross-validation sweep over RISE ranks (`rank_selection.run_bicv`).

```bash
uv run ccc-rise-riserank --dataset alad --ranks 5:51:5 --n-folds 5 \
    --selected-rank 25 --notes "elbow of the held-out block R2X curve"
```

Writes `data/rise_rank_alad.csv` (read by `figureRISErank`) and its sidecar.

### 2. CPD rank selection — `run_cpd_rank.py`

Fits RISE **once** at an already-selected rank and builds one interaction
tensor, then sweeps CP ranks over it with entry-masked imputation CV
(`rank_selection.cp_masked_reconstruction_sweep`). Every rank in the sweep is
therefore scored against identical RISE factors.

```bash
uv run ccc-rise-cpdrank --dataset alad --rise-rank 25 --ranks 1:26 --n-masks 5 \
    --selected-rank 14 --notes "plateau of held-out entry R2X"
```

Writes `data/cpd_rank_alad.csv` (read by `figureCPDrank`) and its sidecar. CP
rank selection is conditional on the RISE rank, which is why `--rise-rank` is
required; it is recorded in the sidecar.

### 3. The pipeline — `run_ccc_rise.py`

The end-to-end decomposition at the ranks the two sweeps justified.

```bash
uv run ccc-rise-run --dataset alad --rise-rank 25 --cp-rank 14
```

Writes `data/bal_alad_factors.nc` and its sidecar. The `.nc` is read by the
figure modules via `figures.common.get_alad_data` /
`factors.load_factors_to_anndata`.

## The CP rank cap

`ccc_rise.validate_cp_rank` rejects `cp_rank > min(tensor.shape)`. The
communication tensor's modes are `(n_conditions, rise_rank, rise_rank,
n_lr_pairs)`, so in practice the constraint is `min(n_conditions, rise_rank)`.

Above that, tensorly's SVD initialization falls back to a full SVD of a mode
unfolding — hundreds of GiB for a realistic tensor — and pads the excess
components with uniform random columns, starting them from noise rather than
from the data. The pipeline runner checks this before the RISE fit, so a bad
rank pair fails in seconds.

## Recording *why* a rank was chosen

The pipeline sidecar carries one `rank_selection` entry per stage:

```yaml
rank_selection:
  rise:
    rank: 25
    method: bicv
    sweep_csv: cellcommunicationpf2/data/rise_rank_alad.csv
    notes: elbow of the held-out block R2X curve
  cpd:
    rank: 14
    method: masked_entry_imputation_cv
    sweep_csv: cellcommunicationpf2/data/cpd_rank_alad.csv
    notes: plateau of held-out entry R2X
```

`notes` fields are the only ones meant to be hand-edited, and a re-run
overwrites them — keep the reasoning in the PR too.

## Source-data provenance

Every sidecar records `source_file` and `source_sha256`, so a result can be
tied to the exact input it was computed from and a silent change in the source
data becomes visible. Hashing a multi-GB source takes a minute or two,
negligible beside the fit it accompanies.

## Typical sequence

```bash
uv run ccc-rise-riserank --dataset alad --ranks 5:51:5 --selected-rank 25 \
    --notes "elbow of the held-out block R2X curve"
uv run ccc-rise-cpdrank  --dataset alad --rise-rank 25 --ranks 1:26 \
    --selected-rank 14 --notes "plateau of held-out entry R2X"
uv run ccc-rise-run      --dataset alad --rise-rank 25 --cp-rank 14
make output/figureRISErank.svg output/figureCPDrank.svg
```
