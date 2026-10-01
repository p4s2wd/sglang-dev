#!/bin/bash
# usage: ab_grid.sh <HB_FAST> <tag>
# A/B the grid axis order for the head-shared prefill attention kernel. Prefill
# only, three salted probes per boot; the kernel changed a lot since this was
# last tried (64-column dots, 8 chunk accumulators, 2 stages), so the old
# "noise" verdict does not carry over.
cd /data/nvme/sglang-codex
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_HS_GRID_HB_FAST=$1 \
  bash ./serve.sh 2 4 gr-$2 --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 60); do
  sleep 10
  grep -qaE "fired up|Error|Traceback" logs/gr-$2.log 2>/dev/null && break
done
grep -qaE "fired up" logs/gr-$2.log || { echo "SERVER FAILED"; exit 1; }
echo "=== HB_FAST=$1 ($2) ==="
for i in 1 2 3; do
  timeout 300 .venv/bin/python probe_prefill_rate2.py 14000 2>&1 | grep -aoE "[0-9.]+ tok/s" | tail -1
done
