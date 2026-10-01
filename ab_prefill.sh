#!/bin/bash
# usage: ab_prefill.sh <NCOL> <WARPS> <tag>
cd /data/nvme/sglang-codex
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_HS_NCOL=$1 SGLANG_SM75_HS_WARPS=$2 \
  bash ./serve.sh 2 4 ab-$3 --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 60); do
  sleep 10
  if grep -qaE "fired up" logs/ab-$3.log 2>/dev/null; then break; fi
done
grep -qaE "fired up" logs/ab-$3.log || { echo "$3 FAILED TO START"; exit 1; }
echo "== $3 (NCOL=$1 WARPS=$2) =="
for t in 1 2 3; do
  timeout 400 .venv/bin/python probe_prefill_rate2.py 8000 "t$t" 2>&1 | tail -1
done
