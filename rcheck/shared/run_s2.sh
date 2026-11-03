#!/usr/bin/env bash
# S2 (cell loss) in the R-shared arm: upstream Scriabin programs + merge on
# each cell subset of rcheck/shared/cell_subsets.py (the cells cccRISE's S2
# refits use), seed 1111. Checkpointed per sample; rerun to resume.
#   MAX_WORKERS=16 MEM=48G rcheck/shared/run_s2.sh alad
set -uo pipefail
cd "$(dirname "$0")/../.."
DATASET=$1
IN=singlecell_cache/shared/$DATASET
OUT=rcheck/out/shared/$DATASET/s2
SEED=${SEED:-1111}
MEM=${MEM:-48G}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 IP_LR_CSV=$IN/lr.csv PYTHONUNBUFFERED=1
mkdir -p $OUT/logs $OUT/meta
st() { echo "$(date '+%F %T') == $*"; }
SCOPE=cccrise-conc-s2-$DATASET-step.scope
capped() {
  systemctl --user stop $SCOPE 2>/dev/null || true
  systemd-run --user --scope -q --unit $SCOPE -p MemoryMax=$MEM -p MemorySwapMax=0 nice -n 10 "$@"
}
trap 'systemctl --user stop $SCOPE 2>/dev/null; exit 143' TERM INT
workers() {
  local load; load=$(cut -d' ' -f1 /proc/loadavg)
  awk -v n="$(nproc --all)" -v l="$load" -v m="${MAX_WORKERS:-16}" -v cap="${MEM%G}" -v per=3 \
    'BEGIN { w = int(n - l - 4); if (w > m) w = m; if (w > int(cap / per)) w = int(cap / per); if (w < 2) w = 2; print w }'
}
for f in $OUT/subsets/*.txt; do
  tag=$(basename $f .txt); meta=$OUT/meta/$tag.rds; run=$OUT/r/$tag
  [ -f $meta ] || Rscript -e "m <- readRDS('$IN/meta.rds'); k <- readLines('$f'); saveRDS(m[rownames(m) %in% k, , drop = FALSE], '$meta')"
  n_samples=$(Rscript -e "cat(length(unique(readRDS('$meta')\$sample)))")
  mkdir -p $run
  if [ "$(ls $run/*.rds 2>/dev/null | wc -l)" -lt "$n_samples" ]; then
    w=$(workers); st "$tag programs seed $SEED ($w workers)"
    capped Rscript rcheck/faithful/ip_runs.R programs $IN/lr_data.rds $meta $run $SEED 1 $w >> $OUT/logs/$tag.log 2>&1
  fi
  [ "$(ls $run/*.rds | wc -l)" -eq "$n_samples" ] || { st "INCOMPLETE $tag"; continue; }
  if [ ! -f $run/merged.json ]; then
    st "$tag merge"
    capped Rscript rcheck/faithful/merge_sig.R merge $run >> $OUT/logs/$tag.log 2>&1
  fi
done
st done
