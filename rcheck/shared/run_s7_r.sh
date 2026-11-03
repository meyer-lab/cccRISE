#!/usr/bin/env bash
# S7's R reruns (CONCORDANCE_PLAN.md §5.1): upstream Scriabin (patched_ip.R
# via IP_SETTINGS) on the R-shared inputs at the port grid's best settings,
# seeds 1111 2 3, programs + merge (no significance, §8). Checkpointed per
# sample; rerun to resume.
#   rcheck/shared/run_s7_r.sh DATASET SETTINGS_JSON [META_RDS]
# SETTINGS_JSON: {"label": {"r2_cutoff":..,"max_soft_power":..|null,
#   "min_size":..,"tree_cut_quantile":..,"n_iterate":..|null,"merge_j":..}, ...}
# META_RDS: a sample subset (default: every sample). IBD merges in the port.
set -uo pipefail
cd "$(dirname "$0")/../.."
DATASET=$1; SETTINGS=$2; META=${3:-singlecell_cache/shared/$DATASET/meta.rds}
IN=singlecell_cache/shared/$DATASET
OUT=rcheck/out/shared/$DATASET/s7_r
SEEDS=${SEEDS:-1111 2 3}
MEM=${MEM:-48G}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 IP_LR_CSV=$IN/lr.csv PYTHONUNBUFFERED=1
mkdir -p $OUT/logs
st() { echo "$(date '+%F %T') == $*"; }
SCOPE=cccrise-conc-s7r-$DATASET-step.scope
capped() {
  systemctl --user stop $SCOPE 2>/dev/null || true
  systemd-run --user --scope -q --unit $SCOPE -p MemoryMax=$MEM -p MemorySwapMax=0 nice -n 10 "$@"
}
trap 'systemctl --user stop $SCOPE 2>/dev/null; exit 143' TERM INT
workers() {
  local load; load=$(cut -d' ' -f1 /proc/loadavg)
  awk -v n="$(nproc --all)" -v l="$load" -v m="${MAX_WORKERS:-16}" -v cap="${MEM%G}" -v per="${MEM_PER_WORKER:-3}" \
    'BEGIN { w = int(n - l - 4); if (w > m) w = m; if (w > int(cap / per)) w = int(cap / per); if (w < 2) w = 2; print w }'
}
n_samples=$(Rscript -e 'cat(length(unique(readRDS("'$META'")$sample)))')
for label in $(.venv/bin/python -c "import json,sys; print(' '.join(json.load(open('$SETTINGS'))))"); do
  cfg=$(.venv/bin/python -c "import json; d=json.load(open('$SETTINGS'))['$label']; print(json.dumps({k: v for k, v in d.items() if k != 'merge_j'}))")
  thr=$(.venv/bin/python -c "import json; print(json.load(open('$SETTINGS'))['$label']['merge_j'])")
  for seed in $SEEDS; do
    run=$OUT/$label/seed$seed
    mkdir -p $run
    if [ "$(ls $run/*.rds 2>/dev/null | wc -l)" -lt "$n_samples" ]; then
      w=$(workers); st "$label programs seed $seed ($w workers) $cfg"
      IP_SETTINGS=$cfg capped Rscript rcheck/faithful/ip_runs.R programs $IN/lr_data.rds $META $run $seed 1 $w \
        >> $OUT/logs/$label.seed$seed.log 2>&1
    fi
    n=$(ls $run/*.rds 2>/dev/null | wc -l)
    [ "$n" -eq "$n_samples" ] || { st "INCOMPLETE $label seed $seed: $n of $n_samples"; continue; }
    if [ ! -f $run/merged.json ]; then
      st "$label merge seed $seed (threshold $thr)"
      if [ "$DATASET" = ibd ]; then
        capped .venv/bin/python rcheck/shared/merge_port.py $run $thr >> $OUT/logs/$label.seed$seed.log 2>&1
      else
        capped Rscript rcheck/faithful/merge_sig.R merge $run $thr >> $OUT/logs/$label.seed$seed.log 2>&1
      fi
    fi
  done
done
st done
