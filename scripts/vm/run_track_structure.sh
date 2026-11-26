#!/bin/bash
# Level curves and sections for every held track, both sources, 2026-11-13. Resumable: rerun the same script.
# Pulls the tracks from S3, measures, syncs the results to S3 and STOPS THE VM at the end.
cd ~/AI-DJ
P=~/miniconda3/envs/aidj/bin/python
B=s3://aidj-telfy/djdata; D=s3://aidj-telfy/djdata_djs
L=logs/track_structure_2026-11-13.log
mkdir -p logs
echo "=== start $(date -u +%T)" | tee -a $L
aws s3 sync $B/tracks/ data/djdata/raveform/tracks/ --only-show-errors
aws s3 sync $D/tracks/ data/djdata/djs/tracks/ --only-show-errors
echo "tracks on disk: raveform $(ls data/djdata/raveform/tracks | wc -l) djs $(ls data/djdata/djs/tracks | wc -l) $(date -u +%T)" | tee -a $L
( while true; do echo "$(date -u +%T) $(free -m | awk '/Mem/{print "used_mb",$3,"avail_mb",$7}') load $(cut -d' ' -f1 /proc/loadavg)"; sleep 60; done ) > logs/track_structure_memory_2026-11-13.log 2>&1 &
MEM=$!
PYTHONPATH=. $P scripts/vm/track_structure.py --workers "${WORKERS:-$(nproc)}" 2>>logs/track_structure_stderr_2026-11-13.log | tee -a $L
kill $MEM
for C in raveform:$B djs:$D; do
  S=${C%%:*}; R=${C#*:}
  aws s3 sync data/djdata/$S/curves/ $R/curves/ --only-show-errors
  aws s3 sync data/djdata/$S/segments/ $R/segments/ --only-show-errors
  aws s3 cp data/djdata/$S/out/track_structure_log.csv $R/out/track_structure_log.csv --only-show-errors
done
echo "SYNCED $(date -u +%T) curves $(ls data/djdata/*/curves | wc -l) segments $(ls data/djdata/*/segments | wc -l)" | tee -a $L
aws s3 cp $L $B/logs/ --only-show-errors
echo "ALL DONE $(date -u +%T)" | tee -a $L
sudo shutdown -h +2
