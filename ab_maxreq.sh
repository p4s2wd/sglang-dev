#!/bin/bash
# usage: ab_maxreq.sh <MAXREQ> <tag>
# Device time is 345 ms per 512-token chunk (1484 tok/s if the GPU were packed)
# but wall time at MAXREQ=2 is 581 ms/chunk (880 tok/s) -- a 40% gap. With PP=4
# the stages can only overlap when more than one request is in flight, so the
# suspect is pipeline serialization, not kernel speed. Same kernels both boots;
# only MAXREQ changes.
cd /data/nvme/sglang-codex
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=$1 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 2 4 mr-$2 --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 60); do
  sleep 10
  grep -qaE "fired up|Error|Traceback" logs/mr-$2.log 2>/dev/null && break
done
grep -qaE "fired up" logs/mr-$2.log || { echo "SERVER FAILED"; grep -aE "Error|Traceback" logs/mr-$2.log | tail -3 | cut -c1-140; exit 1; }
echo "=== MAXREQ=$1 ($2) ==="
grep -aoE "max_total_num_tokens=[0-9]+" logs/mr-$2.log | tail -1
for i in 1 2 3; do
  timeout 300 .venv/bin/python probe_prefill_rate2.py 14000 2>&1 | grep -aoE "[0-9.]+ tok/s" | tail -1
done
