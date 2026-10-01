#!/bin/bash
# usage: ab_one.sh <TAG> <ENV=VAL> [ENV=VAL...]
# Run one arm of the long-context decode A/B.
cd /data/nvme/sglang-codex
. ./env.sh
TAG=$1; shift
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 50
for i in $(seq 1 15); do L=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$L" = "0" ] && break; sleep 10; done
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 "$@" \
  bash ./serve.sh 2 4 ab-$TAG --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 70); do sleep 10; grep -qaE "fired up|out of memory|Traceback" logs/ab-$TAG.log 2>/dev/null && break; done
grep -qaE "fired up" logs/ab-$TAG.log || { echo "$TAG SERVER FAILED"; grep -aE "out of memory|Traceback" logs/ab-$TAG.log|tail -2|cut -c1-140; exit 1; }
echo "=== $TAG ($*) ==="
timeout 1500 .venv/bin/python dec_ctx_ab.py 2>&1 | grep -aE "ctx|FAIL"
timeout 400 .venv/bin/python bs8_correct.py 2>&1 | grep -a "bs=8 batched"
echo DONE
