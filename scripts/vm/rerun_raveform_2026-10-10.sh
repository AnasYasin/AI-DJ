#!/bin/bash
# Raveform only: measure (exit fix, band carried levels), label, types, export, S3 sync. Started 2026-10-10 beside
# the DJ-corpus locate stage at 8 workers, then stopped (19 GB used, no headroom) and chained after the DJ stages
# instead, at 14 workers as in the 2026-09-30 run. Called from rerun_placement_2026-10-09.sh (VM copy).
# The Raveform block was removed from the VM copy of rerun_placement_2026-10-09.sh so the two never write one table.
set -euo pipefail
cd ~/AI-DJ
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aidj
export PYTHONPATH=djdata_package:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STAMP=before_2026-10-09
run() { echo "== $(date -u +%H:%M:%S) $*"; "$@"; }
C=djdata_package/config.yaml
OUT=data/djdata/raveform/out
mkdir -p "$OUT/$STAMP"
for f in measures.csv labels.csv types.csv seams_index.csv; do
  if [ -e "$OUT/$f" ] && [ ! -e "$OUT/$STAMP/$f" ]; then mv "$OUT/$f" "$OUT/$STAMP/$f"; fi
done
ls -la "$OUT/$STAMP"
run python -m djdata.cli measure      --config $C --workers 16   # Raveform windows are short (max 395 s), 16 is safe; 2026-10-11
run python -m djdata.cli label        --config $C
run python -m djdata.cli types        --config $C
run python -m djdata.cli export-seams --config $C
run aws s3 sync data/djdata/raveform/out/ s3://aidj-1/djdata/out/
echo "== $(date -u +%H:%M:%S) raveform done"
