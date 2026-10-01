#!/bin/bash
# usage: ab_power.sh <W> <tag>
# A/B the power limit WITHOUT restarting the server. nvidia-smi's power limit is
# applied dynamically, so the same running server, the same compiled kernels, the
# same KV pool and the same thermal history can be measured at 150 W and at 250 W
# back to back. That removes every confound that has plagued these comparisons
# (reboot, recompile, session drift) -- and the drift itself is what we are
# measuring, since a 150 W cap makes the clock float with ambient conditions.
#
# Interleaved: 150, 250, 150, 250. If 250's worst beats 150's best, the effect is
# real by the standing rule.
cd /data/nvme/sglang-codex
W=$1; TAG=$2
sudo -n nvidia-smi -pl $W >/dev/null 2>&1
echo "all GPUs set to $W W:"
nvidia-smi --query-gpu=power.limit --format=csv,noheader | tr '\n' ' '; echo
timeout 1300 .venv/bin/python pf_curve.py 8000,12000 3 2>&1 | tail -5
echo "--- decode at $W W ---"
timeout 400 bash probe_perf.sh 3 64 2>&1 | grep -a "decode throughput"
