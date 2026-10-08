#!/bin/bash
# Re-run the seam pipeline with the 2026-10-09 code on both corpora, from the repo root on the VM.
#
#   DJ corpus:  locate again (tempo-refined, key-lock aware placement), then every stage after it
#   Raveform:   placement is Raveform's own, so measure (exit fix, band carried levels), label, types, export
#
# The old tables are moved to out/before_2026-10-09/ first, never deleted. Stops at the first failure,
# leaves the VM on; Anas stops it. Run inside tmux: bash scripts/vm/rerun_placement_2026-10-09.sh
set -euo pipefail
cd ~/AI-DJ
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aidj
export PYTHONPATH=djdata_package:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STAMP=before_2026-10-09
run() { echo "== $(date -u +%H:%M:%S) $*"; "$@"; }

keep() {  # move the tables a stage will rewrite into the dated folder, keep what stays valid
  local out=$1; shift
  mkdir -p "$out/$STAMP"
  # a second start keeps the first run's dated copy and leaves the live, partly filled table in place
  # (coreutils 9 makes `mv -n` fail when it skips, which killed the script under set -e, 2026-10-09)
  for f in "$@"; do
    if [ -e "$out/$f" ] && [ ! -e "$out/$STAMP/$f" ]; then mv "$out/$f" "$out/$STAMP/$f"; fi
  done
  ls -la "$out/$STAMP" | head -20
}

# ---- DJ corpus: placement and everything after it ------------------------------------------------
C=djdata_package/config_djs.yaml
OUT=data/djdata/djs/out
keep $OUT plays.csv presence.csv mixes.csv seams.csv cuts.csv measures.csv labels.csv types.csv seams_index.csv layers.csv layer_bands.csv
# tempos.csv stays: the record BPMs are the locate refinement's input and did not change
# the windows were cut around the old placement; the cut stage adopts an existing window by seam id,
# so they are set aside too and every seam is cut again from the mix audio (a stream copy, minutes)
if [ -d data/djdata/djs/windows ] && [ ! -d data/djdata/djs/windows_$STAMP ]; then
  mv data/djdata/djs/windows data/djdata/djs/windows_$STAMP
fi
mkdir -p data/djdata/djs/windows
# a second start resumes: `mv -n` above keeps the first run's dated copies, every stage skips its done rows
run python -m djdata.cli locate       --config $C --workers 16
run python -m djdata.cli pairs        --config $C
run python -m djdata.cli cut          --config $C --workers 16
run python -m djdata.cli measure      --config $C --workers 16
run python -m djdata.cli label        --config $C
run python -m djdata.cli types        --config $C
run python -m djdata.cli export-seams --config $C
run python -m djdata.cli layers       --config $C
run python -m djdata.cli layer-bands  --config $C --workers 16

# ---- Raveform: measure and after -------------------------------------------------------------------
C=djdata_package/config.yaml
OUT=data/djdata/raveform/out
keep $OUT measures.csv labels.csv types.csv seams_index.csv
run python -m djdata.cli measure      --config $C --workers 16
run python -m djdata.cli label        --config $C
run python -m djdata.cli types        --config $C
run python -m djdata.cli export-seams --config $C

# ---- to S3, dated copies beside the live tables ---------------------------------------------------
run aws s3 sync data/djdata/djs/out/      s3://aidj-1/djdata_djs/out/
run aws s3 sync data/djdata/raveform/out/ s3://aidj-1/djdata/out/
echo "== $(date -u +%H:%M:%S) done"
