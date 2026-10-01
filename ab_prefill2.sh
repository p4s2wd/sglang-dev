#!/bin/bash
# usage: ab_prefill2.sh <NCOL> <WARPS> <STAGES> <tag>
cd /data/nvme/sglang-codex
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_HS_NCOL=$1 SGLANG_SM75_HS_WARPS=$2 \
  SGLANG_SM75_HS_STAGES=$3 \
  bash ./serve.sh 2 4 ab2-$4 --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 60); do
  sleep 10
  grep -qaE "fired up" logs/ab2-$4.log 2>/dev/null && break
done
grep -qaE "fired up" logs/ab2-$4.log || { echo "$4 FAILED"; exit 1; }
echo "== $4 (NCOL=$1 WARPS=$2 STAGES=$3) =="
for t in 1 2 3; do
  timeout 400 .venv/bin/python probe_prefill_rate2.py 8000 "t$t" 2>&1 | tail -1
done
