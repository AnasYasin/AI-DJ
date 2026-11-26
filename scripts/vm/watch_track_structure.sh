#!/bin/bash
# Watches the track structure run (run_track_structure.sh in tmux session "structure") once a minute and
# restarts it when it is in trouble. The run is resumable, so a restart loses only the tracks in flight.
#   memory: under 8 GB available  -> restart with 8 fewer workers
#   stall:  no new segment file for 15 minutes while the run is alive -> restart, same workers
# At most 3 restarts, then it stops watching and says so. Every action goes to logs/watchdog_track_structure.log.
cd ~/AI-DJ
W=logs/watchdog_track_structure.log
workers=${WORKERS:-$(nproc)}
restarts=0
last_n=-1
last_change=$(date +%s)
say() { echo "$(date -u +%F\ %T) $*" | tee -a $W; }
restart() {
  restarts=$((restarts + 1))
  if [ $restarts -gt 3 ]; then say "GAVE UP after 3 restarts, run left as it is"; exit 1; fi
  tmux kill-session -t structure 2>/dev/null
  sleep 3
  pkill -9 -f "scripts/vm/[t]rack_structure.py"
  sleep 2
  rm -f data/djdata/*/curves/*.tmp.npz data/djdata/*/segments/*.tmp.json
  tmux new-session -d -s structure "WORKERS=$workers bash ~/AI-DJ/scripts/vm/run_track_structure.sh"
  last_change=$(date +%s)
  say "RESTARTED ($restarts of 3) with $workers workers"
}
say "watching, $workers workers"
while true; do
  sleep 60
  if grep -q "ALL DONE" logs/track_structure_2026-11-13.log 2>/dev/null; then say "run finished"; exit 0; fi
  tmux has-session -t structure 2>/dev/null || { say "RUN GONE without finishing"; restart; continue; }
  pgrep -f "scripts/vm/[t]rack_structure.py" > /dev/null || { last_change=$(date +%s); continue; }  # downloading or uploading
  avail=$(free -m | awk '/Mem/{print $7}')
  n=$(ls data/djdata/raveform/segments data/djdata/djs/segments 2>/dev/null | wc -l)
  [ "$n" != "$last_n" ] && { last_n=$n; last_change=$(date +%s); }
  if [ "$avail" -lt 8000 ]; then
    workers=$((workers - 8))
    say "LOW MEMORY ${avail} MB available at $n segments"
    restart
  elif [ $(($(date +%s) - last_change)) -gt 900 ]; then
    say "STALLED at $n segments for 15 minutes"
    restart
  fi
done
