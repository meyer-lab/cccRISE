#!/usr/bin/env bash
# S1 (seed) in the R-shared arm: upstream Scriabin on cccRISE's inputs, per
# sample, for each seed (CONCORDANCE_PLAN.md §5.1, §10 freeze of 2026-11-02):
# programs for seeds 1111 and 2-10, each seed's FindAllInteractionPrograms
# merge, and significance (10k replicates) + module scores for 1111 and 2.
# Every step checkpoints (per sample where it can) and skips finished
# outputs, so the script can be rerun after a crash.
#
#   bash rcheck/shared/run_s1.sh alad      # from the repo root
# Subsamples are stratified by the export's `celltype` (broad_cell_type).
set -euo pipefail
DATASET=$1
IN=singlecell_cache/shared/$DATASET
OUT=rcheck/out/shared/$DATASET/s1
SEEDS=${SEEDS:-"1111 2 3 4 5 6 7 8 9 10"}
SIG_SEEDS=${SIG_SEEDS:-"1111 2"}
MEM=${MEM_CAP:-48G}
# Merge engine: upstream R (default), or the port's exact incremental merge
# (MERGE=port; IBD, where the R merge takes days per seed)
MERGE=${MERGE:-r}
# Significance engine: upstream R (default), or none (skip: IBD, pending a
# decision; R is ~1,400 CPU-h per seed there)
SIG_ENGINE=${SIG_ENGINE:-r}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 IP_LR_CSV=$IN/lr.csv
mkdir -p $OUT/logs
# Each step in its own named scope, stopped with the script (stopping the
# service alone leaves a scope's processes running)
SCOPE=""
capped() {
  SCOPE=cccrise-conc-$DATASET-step.scope
  systemctl --user stop $SCOPE 2>/dev/null || true
  systemd-run --user --scope -q --unit $SCOPE -p MemoryMax=$MEM -p MemorySwapMax=0 nice -n 10 "$@"
}
trap '[ -n "$SCOPE" ] && systemctl --user stop $SCOPE; exit 143' TERM INT
step() { echo "$(date +%H:%M:%S) == $*"; }
# Workers: leave 4 cores spare beyond current load (nproc --all: plain nproc
# honours OMP_NUM_THREADS=1), at most MEM_CAP / MEM_PER_WORKER (GB; IBD
# 2,000-cell samples peak at 4.8 GB per worker), within 2..MAX_WORKERS
workers() {
  local load; load=$(cut -d' ' -f1 /proc/loadavg)
  awk -v n="$(nproc --all)" -v l="$load" -v m="${MAX_WORKERS:-16}"       -v cap="${MEM%G}" -v per="${MEM_PER_WORKER:-5}"     'BEGIN { w = int(n - l - 4); if (w > m) w = m; if (w > int(cap / per)) w = int(cap / per); if (w < 2) w = 2; print w }'
}
n_samples=$(Rscript -e 'cat(length(unique(readRDS("'$IN'/meta.rds")$sample)))')

for seed in $SEEDS; do
  run=$OUT/seed$seed
  if [ "$(ls $run/*.rds 2>/dev/null | wc -l)" -lt "$n_samples" ]; then
    w=$(workers); step "programs seed $seed ($w workers)"
    capped Rscript rcheck/faithful/ip_runs.R programs $IN/lr_data.rds $IN/meta.rds $run $seed 1 $w \
      >> $OUT/logs/seed$seed.log 2>&1
  fi
  n=$(ls $run/*.rds | wc -l)
  [ "$n" -eq "$n_samples" ] || { step "INCOMPLETE seed $seed: $n of $n_samples"; continue; }
  if [ ! -f $run/merged.json ]; then
    step "merge seed $seed ($MERGE)"
    if [ "$MERGE" = port ]; then
      capped .venv/bin/python rcheck/shared/merge_port.py $run >> $OUT/logs/seed$seed.log 2>&1
    else
      capped Rscript rcheck/faithful/merge_sig.R merge $run >> $OUT/logs/seed$seed.log 2>&1
    fi
  fi
done

[ "$SIG_ENGINE" = none ] && { step "significance skipped (SIG_ENGINE=none)"; step done; exit 0; }
for seed in $SIG_SEEDS; do
  run=$OUT/seed$seed
  [ -f $run/merged.json ] || continue
  if [ ! -f $run/kept.json ]; then
    w=$(workers); step "significance seed $seed ($w workers)"
    capped Rscript rcheck/faithful/merge_sig.R significance $run $w >> $OUT/logs/seed$seed.log 2>&1
  fi
  if [ -f $run/kept.json ] && [ ! -f $run/scores_receptors.csv ] && [ -f $IN/full_data.rds ]; then
    step "score seed $seed"
    capped Rscript rcheck/faithful/merge_sig.R score $run $IN/full_data.rds $IN/meta.rds >> $OUT/logs/seed$seed.log 2>&1
  fi
done
step done
