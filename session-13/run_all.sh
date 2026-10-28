#!/bin/zsh
# the three Session 13 runs, one after another
set -e
python3 rev_train.py probe
MAXB=$(python3 -c "import json; print(json.load(open('logs/memory_probe.json'))['max_batch']['midpoint'])")
python3 rev_train.py train standard standard 32
python3 rev_train.py train reversible midpoint 32
python3 rev_train.py train rev_maxbatch midpoint $MAXB
