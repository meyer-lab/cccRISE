#!/usr/bin/env bash
# S7 port grid (rcheck/shared/s7_grid.py) for ALAD then IBD, as a user
# service. Idempotent: every stage resumes from its checkpoints.
#   DATASETS="alad ibd" MAX_WORKERS=16 MEM=48G rcheck/shared/run_s7.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
MEM=${MEM:-48G}
st() { echo "$(date '+%F %T') == $*"; }
workers() {
  local load; load=$(cut -d' ' -f1 /proc/loadavg)
  awk -v n="$(nproc --all)" -v l="$load" -v m="${MAX_WORKERS:-16}" -v cap="${MEM%G}" -v per=2 \
    'BEGIN { w = int(n - l - 4); if (w > m) w = m; if (w > int(cap / per)) w = int(cap / per); if (w < 2) w = 2; print w }'
}
capped() {
  systemd-run --user --scope -q --unit "cccrise-conc-s7-step-$1.scope" \
    -p MemoryMax=$MEM -p MemorySwapMax=0 nice -n 10 "${@:2}"
}
trap 'systemctl --user stop "cccrise-conc-s7-step-*.scope" 2>/dev/null; exit 143' TERM
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
for ds in ${DATASETS:-alad ibd}; do
  for stage in programs merge agree; do
    w=$(workers); st "$ds $stage ($w workers)"
    capped "$ds-$stage" .venv/bin/python rcheck/shared/s7_grid.py $stage --dataset $ds --workers $w \
      || { st "$ds $stage FAILED"; exit 1; }
  done
done
st done
