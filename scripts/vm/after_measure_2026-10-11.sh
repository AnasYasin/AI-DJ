#!/bin/bash
# The DJ-corpus stages after measure, then Raveform, then the S3 sync. Written 2026-10-11 after the measure stage
# had to run alone (35 windows over 1,500 s set aside in windows_big/, list in out/measure_big_windows_2026-10-11.csv;
# their rows are missing until Anas decides). label/types/export rebuild whole, so they run again after that decision.
set -euo pipefail
cd ~/AI-DJ
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aidj
export PYTHONPATH=djdata_package:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { echo "== $(date -u +%H:%M:%S) $*"; "$@"; }
C=djdata_package/config_djs.yaml
run python -m djdata.cli label        --config $C
run python -m djdata.cli types        --config $C
run python -m djdata.cli export-seams --config $C
run python -m djdata.cli layers       --config $C
run python -m djdata.cli layer-bands  --config $C --workers 16
run bash scripts/vm/rerun_raveform_2026-10-10.sh
run aws s3 sync data/djdata/djs/out/      s3://aidj-1/djdata_djs/out/
run aws s3 sync data/djdata/raveform/out/ s3://aidj-1/djdata/out/
echo "== $(date -u +%H:%M:%S) done"
