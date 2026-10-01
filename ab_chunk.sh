#!/bin/bash
# usage: ab_chunk.sh <CHUNK> <tag>
# Per-stage device time is 345 ms per 512-token chunk while wall time is 582 ms,
# and the PP stages are balanced to 1.07x, so the gap is per-step overhead and
# pipeline fill rather than an unbalanced stage. Larger chunks amortize that
# overhead over more tokens. Chunk 512->1024 was rejected in an earlier round,
# but that was when attention was 39.6% of device time and far slower; with the
# kernels now faster, fixed per-step costs are relatively larger, so the old
# verdict needs re-measuring rather than trusting.
cd /data/nvme/sglang-codex
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=$1 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 2 4 ck-$2 --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 60); do
  sleep 10
  grep -qaE "fired up|Error|Traceback" logs/ck-$2.log 2>/dev/null && break
done
grep -qaE "fired up" logs/ck-$2.log || { echo "SERVER FAILED"; grep -aE "Error|Traceback" logs/ck-$2.log | tail -3 | cut -c1-140; exit 1; }
echo "=== CHUNK=$1 ($2) ==="
for i in 1 2 3; do
  timeout 300 .venv/bin/python probe_prefill_rate2.py 14000 2>&1 | grep -aoE "[0-9.]+ tok/s" | tail -1
done
bash ./probe_perf.sh 3 64 2>&1 | grep -a "decode throughput"
