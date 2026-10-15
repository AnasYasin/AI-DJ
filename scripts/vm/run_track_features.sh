#!/bin/bash
# Full track-feature run, both corpora, 2026-10-07. One table per corpus. Resumable: rerun the same script.
# Syncs to S3 and STOPS THE VM at the end (unlike the seam runs, nothing follows this unattended).
cd ~/AI-DJ
P=~/miniconda3/envs/aidj/bin/python
L=logs/track_features_2026-10-07.log
( while true; do echo "$(date -u +%T) $(free -m | awk '/Mem/{print "used_mb",$3,"avail_mb",$7}') load $(cut -d' ' -f1 /proc/loadavg)"; sleep 60; done ) > logs/track_features_memory_2026-10-07.log 2>&1 &
MEM=$!
for C in raveform djs; do
  echo "=== $C start $(date -u +%T)" | tee -a $L
  OMP_NUM_THREADS=2 PYTHONPATH=. $P -m src.features.track_features \
    --tracks $C=data/djdata/$C/tracks --tempos $C=data/djdata/$C/out/tempos.csv \
    --out data/djdata/$C/out/track_features.parquet --embeddings data/djdata/$C/embeddings \
    --workers 6 2>>logs/track_features_stderr_2026-10-07.log | grep --line-buffered -E "INFO" >> $L
  echo "=== $C end $(date -u +%T)" | tee -a $L
done
kill $MEM
B=s3://aidj-1/djdata; D=s3://aidj-1/djdata_djs
aws s3 cp data/djdata/raveform/out/track_features.parquet $B/out/track_features.parquet --only-show-errors
aws s3 sync data/djdata/raveform/embeddings/ $B/embeddings/ --only-show-errors
aws s3 cp data/djdata/djs/out/track_features.parquet $D/out/track_features.parquet --only-show-errors
aws s3 sync data/djdata/djs/embeddings/ $D/embeddings/ --only-show-errors
aws s3 cp $L $B/logs/ --only-show-errors
echo "SYNCED $(date -u +%T) raveform $(ls data/djdata/raveform/embeddings | wc -l) djs $(ls data/djdata/djs/embeddings | wc -l)" | tee -a $L
echo "ALL DONE $(date -u +%T)" | tee -a $L
sudo shutdown -h +2
