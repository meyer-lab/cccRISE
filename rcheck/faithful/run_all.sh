#!/usr/bin/env bash
# Faithful Scriabin on ALAD, end to end (see SCRIABIN_FAITHFUL_REPORT.md).
# Every step checkpoints (per sample where it can) and skips finished
# outputs, so the script can be rerun after a crash. Each step runs in its
# own memory-capped scope (shared machine).
#
#   bash rcheck/faithful/run_all.sh            # from the repo root
set -euo pipefail
F=rcheck/out/faithful
W=${WORKERS:-6}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p $F/logs $F/runs
capped() { local mem=$1; shift; systemd-run --user --scope -q -p MemoryMax=$mem -p MemorySwapMax=0 nice -n 10 "$@"; }
step() { echo "$(date +%H:%M:%S) == $*"; }

# 1. preprocessing: full depth, and UMIs downsampled (inDrop depth = 5828 / 11282 mean UMIs)
for p in 1 0.5166 0.25 0.1; do
  d=$F/depth$p; [ "$p" = 0.5166 ] && d=$F/depth0.52
  [ -f $d/meta.rds ] && continue
  step preprocess $p
  extra=""; [ "$p" = 1 ] && extra=full
  capped 40G Rscript rcheck/faithful/preprocess.R $F/input $d $p $extra > $F/logs/preprocess_$(basename $d).log 2>&1
done

# 2. per-sample interaction programs: run name, data, seed, fraction of cells
programs() {
  local run=$1 data=$2 seed=$3 frac=${4:-1}
  step programs $run
  capped 40G Rscript rcheck/faithful/ip_runs.R programs $data $F/depth1/meta.rds $F/runs/$run $seed $frac $W \
    >> $F/logs/$run.log 2>&1
  local n; n=$(ls $F/runs/$run/*.rds | wc -l)
  [ "$n" -eq 47 ] || { step "INCOMPLETE $run: $n of 47 samples, not merged"; return 0; }
  [ -f $F/runs/$run/merged.json ] || capped 8G Rscript rcheck/faithful/merge_sig.R merge $F/runs/$run >> $F/logs/$run.log 2>&1
}
significance() {
  [ -f $F/runs/$1/merged.json ] || return 0
  step significance $1
  [ -f $F/runs/$1/kept.json ] || capped 40G Rscript rcheck/faithful/merge_sig.R significance $F/runs/$1 $W >> $F/logs/$1.log 2>&1
}

programs alra_s1111 $F/depth1/alra_lr.rds 1111
significance alra_s1111
[ ! -f $F/runs/alra_s1111/kept.json ] || [ -f $F/runs/alra_s1111/scores_receptors.csv ] || { step score alra_s1111; capped 40G Rscript rcheck/faithful/merge_sig.R score \
  $F/runs/alra_s1111 $F/depth1/alra_full.rds $F/depth1/meta.rds >> $F/logs/alra_s1111.log 2>&1; }
programs alra_s2 $F/depth1/alra_lr.rds 2
significance alra_s2
programs alra_s3 $F/depth1/alra_lr.rds 3
programs sct_s1111 $F/depth1/sct_lr.rds 1111
significance sct_s1111
programs sct_s2 $F/depth1/sct_lr.rds 2
programs alra_f50_s1111 $F/depth1/alra_lr.rds 1111 0.5
for d in 0.52 0.25 0.1; do programs alra_d${d}_s1111 $F/depth$d/alra_lr.rds 1111; done

# 3. one cell left out, no randomness (full CCIM): samples of <= 300 cells
step cell drop
for s in $(Rscript -e 'm <- readRDS("'$F'/depth1/meta.rds"); n <- table(m$sample); cat(names(n)[n <= 300])'); do
  capped 16G Rscript rcheck/faithful/ip_runs.R drop $F/depth1/alra_lr.rds $F/depth1/meta.rds $F/runs/drop $s 0 37 74 \
    >> $F/logs/drop.log 2>&1
done

# 4. the paper's CCIM noise analysis (LISI), three smallest samples
mkdir -p $F/lisi
for s in $(Rscript -e 'm <- readRDS("'$F'/depth1/meta.rds"); n <- sort(table(m$sample)); cat(names(n)[1:3])'); do
  [ -f $F/lisi/$s.csv ] && continue
  step lisi $s
  capped 32G Rscript rcheck/faithful/ccim_lisi.R $F/depth1 $F/lisi/$s.csv $s $F/depth0.52 $F/depth0.25 $F/depth0.1 \
    >> $F/logs/lisi.log 2>&1
done
step done
