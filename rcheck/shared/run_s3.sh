#!/usr/bin/env bash
# S3 (one cell, no randomness) in the R-shared arm: for each sample in
# rcheck/out/shared/<ds>/s3/drops.csv (rcheck/shared/cell_subsets.py), upstream
# InteractionPrograms on the full CCIM with all cells and with each listed cell
# left out (ip_runs.R drop). Samples run in parallel (P at a time); each run
# is its own checkpoint (<sample>_drop<idx>.rds), so rerun to resume.
#   P=4 MEM=48G rcheck/shared/run_s3.sh alad
set -uo pipefail
cd "$(dirname "$0")/../.."
DATASET=$1
IN=singlecell_cache/shared/$DATASET
OUT=rcheck/out/shared/$DATASET/s3/r
mkdir -p $OUT/logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 IP_LR_CSV=$IN/lr.csv IN OUT DATASET
SCOPE=cccrise-conc-s3-$DATASET.scope
trap 'systemctl --user stop $SCOPE 2>/dev/null; exit 143' TERM INT
echo "$(date '+%F %T') == S3 $DATASET, ${P:-4} at a time"
systemd-run --user --scope -q --unit $SCOPE -p MemoryMax=${MEM:-48G} -p MemorySwapMax=0 nice -n 10 \
  bash -c '
  tail -n +2 $OUT/../drops.csv | awk -F, "{a[\$1]=a[\$1]\" \"\$3} END {for (s in a) print s a[s]}" |
    xargs -P '"${P:-4}"' -L 1 bash -c '"'"'Rscript rcheck/faithful/ip_runs.R drop $IN/lr_data.rds $IN/meta.rds $OUT "$0" "$@" > $OUT/logs/$0.log 2>&1; echo "$(date +%T) $0 exit $?"'"'"'
  '
echo "$(date '+%F %T') == done ($(ls $OUT/*.rds | wc -l) runs)"
