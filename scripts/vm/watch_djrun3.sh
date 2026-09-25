#!/bin/bash
# Watch the track fetch on the VM. Idle = no "track done" / "track FAILED" log event for 20 min, or no
# pending+downloading tracks, or the run dead on two ticks. Then stop the run, copy results, stop the VM.
SP=$(cd "$(dirname "$0")" && pwd)
cd /home/anas-yasin/projects/AI-DJ/AI-DJ
sleep 300
dead_ticks=0
for i in $(seq 1 200); do
  out=$($SP/vmssh 'cd ~/AI-DJ && ~/miniconda3/envs/aidj/bin/python -c "
import sqlite3; c=sqlite3.connect(\"data/djdata/djs/state.sqlite\")
print(dict(c.execute(\"select status, count(*) from tracks group by status\").fetchall()))"; last=$(grep -E "track done|track FAILED" data/djdata/djs/logs/djdata.log | tail -1 | cut -c1-19); echo "last_event $last"; echo "idle_s $(( $(date +%s) - $(date -d "$last" +%s 2>/dev/null || date +%s) ))"; tmux has-session -t djrun 2>/dev/null && echo alive || echo dead; python3 -c "import json;d=json.load(open(\"data/djdata/djs/logs/gate.json\"));print(\"gate\",d[\"state\"],\"blocks\",d[\"blocks\"],\"rotations\",d[\"rotations\"])"' 2>&1)
  echo "$(date +%H:%M) $out" | tr '\n' ' '; echo
  if [ -z "$out" ] || echo "$out" | grep -q "Connection timed out\|no public ip"; then echo "VM unreachable this tick"; sleep 300; continue; fi
  pending=$(echo "$out" | grep -oE "'pending': [0-9]+" | grep -oE "[0-9]+"); downloading=$(echo "$out" | grep -oE "'downloading': [0-9]+" | grep -oE "[0-9]+")
  idle=$(echo "$out" | grep -oE "idle_s [0-9]+" | grep -oE "[0-9]+")
  if echo "$out" | grep -qw dead; then dead_ticks=$((dead_ticks+1)); else dead_ticks=0; fi
  [ $dead_ticks -ge 2 ] && { echo "run dead on two ticks"; break; }
  [ "${pending:-1}" = "0" ] && [ "${downloading:-0}" = "0" ] && { echo "nothing pending: done"; break; }
  [ "${idle:-0}" -gt 1200 ] && { echo "IDLE: no track event for $((idle/60)) min"; break; }
  sleep 300
done
$SP/vmssh 'bash /tmp/stop_run.sh'
rsync -a -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new" aidj:~/AI-DJ/data/djdata/djs/logs/djdata.log logs/djrun_tracks_vm_2026-09-24.log
rsync -a -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new" aidj:~/AI-DJ/data/djdata/djs/state.sqlite data/djdata/djs/state.sqlite.vm-2026-09-24
echo "results copied"
aws ec2 stop-instances --profile talhanonstatic --region us-east-1 --instance-ids i-023e6cd5518feff75 --query 'StoppingInstances[0].CurrentState.Name' --output text
