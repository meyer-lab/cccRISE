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
SEEDS="1111 2 3 4 5 6 7 8 9 10"
SIG_SEEDS="1111 2"
MEM=${MEM_CAP:-48G}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 IP_LR_CSV=$IN/lr.csv
mkdir -p $OUT/logs
capped() { systemd-run --user --scope -q -p MemoryMax=$MEM -p MemorySwapMax=0 nice -n 10 "$@"; }
step() { echo "$(date +%H:%M:%S) == $*"; }
# Workers from current load: leave 4 cores spare, 2..MAX_WORKERS
workers() {
  local load; load=$(cut -d' ' -f1 /proc/loadavg)
  awk -v n="$(nproc)" -v l="$load" -v m="${MAX_WORKERS:-16}" \
    'BEGIN { w = int(n - l - 4); if (w > m) w = m; if (w < 2) w = 2; print w }'
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
  [ -f $run/merged.json ] || { step "merge seed $seed"; capped Rscript rcheck/faithful/merge_sig.R merge $run >> $OUT/logs/seed$seed.log 2>&1; }
done

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
