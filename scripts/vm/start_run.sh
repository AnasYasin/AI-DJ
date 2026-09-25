#!/bin/bash
cd ~/AI-DJ
tmux new -d -s djrun "cd ~/AI-DJ && PATH=$HOME/.deno/bin:\$PATH PYTHONWARNINGS=ignore ~/miniconda3/envs/aidj/bin/djdata run --config djdata_package/config_djs.yaml >> data/djdata/djs/logs/djrun.console 2>&1"
sleep 2; tmux has-session -t djrun 2>/dev/null && echo "run started" || echo "RUN DID NOT START"
