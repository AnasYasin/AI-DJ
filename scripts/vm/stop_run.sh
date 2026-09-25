#!/bin/bash
tmux kill-session -t djrun 2>/dev/null; sleep 3
pkill -f "djdata run --config" 2>/dev/null; sleep 3
echo "run procs left: $(pgrep -fc "djdata run --config" || echo 0)"
