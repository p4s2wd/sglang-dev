#!/bin/bash
# usage: ab_v3p.sh <NT> <tag>  -- prefill A/B at the MAXREQ=2 config prefill was
# characterized at (733 tok/s). MAXREQ=8 adds CUDA-graph buffers for four batch
# sizes on top of a 256K KV pool and OOMs a 14K prefill, so prefill must be
# measured at MAXREQ=2; decode throughput is measured separately at MAXREQ=8.
cd /data/nvme/sglang-codex
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
NT=$1; TAG=$2
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=$NT \
  bash ./serve.sh 2 4 v3p-$TAG --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 60); do
  sleep 10
  grep -qaE "fired up|Error|Traceback" logs/v3p-$TAG.log 2>/dev/null && break
done
grep -qaE "fired up" logs/v3p-$TAG.log || { echo "SERVER FAILED"; exit 1; }
echo "=== prefill NT=$NT ($TAG) ==="
for i in 1 2 3; do
  timeout 300 .venv/bin/python probe_prefill_rate2.py 14000 2>&1 | grep -aoE "[0-9.]+ tok/s" | tail -1
done
curl -s -m 5 http://127.0.0.1:30000/health >/dev/null && echo "server ALIVE after probes" || echo "server DOWN after probes"
