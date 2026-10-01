#!/bin/bash
# usage: clocks_under_load.sh
# power.limit reads 150W against power.max_limit 280-330W, with SM clocks at
# 1350 MHz of a 2145 MHz max. If that cap binds under load, every throughput
# number on this box is measured at ~63% of available clock and raising it is a
# free win. Idle readings prove nothing, so sample every 3 s across the whole
# prefill (a 24.6K-token prompt takes ~25 s) starting immediately.
cd /data/nvme/sglang-codex
setsid nohup timeout 400 .venv/bin/python probe_prefill_rate2.py 14000 >/tmp/pfload.log 2>&1 </dev/null &
PROBE=$!
sleep 8
for i in $(seq 1 12); do
  printf "t=%2ds " $((8 + i * 3))
  nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '{printf "%s/%s/%s%% ", $1, $2, $3} END{print ""}'
  sleep 3
done
wait $PROBE 2>/dev/null
tail -2 /tmp/pfload.log
