#!/bin/bash
# usage: ab_bs.sh <MAXREQ> <GRAPH_BS> <tag>
# Decode is PP4-serialised per token (4 stages x 17.13 ms = 68.5 ms = 14.6 tok/s,
# matching the measured 14.9), but the per-step cost is dominated by weight reads
# that do NOT grow with batch size. So aggregate throughput should scale with
# batch. This measures how close bs=4 gets to the 50 tok/s target.
cd /data/nvme/sglang-codex
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=$1 GRAPH_MAX_BS=4 GRAPH_BS="$2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton \
  bash ./serve.sh 2 4 bs-$3 --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 70); do
  sleep 10
  grep -qaE "fired up" logs/bs-$3.log 2>/dev/null && break
done
grep -qaE "fired up" logs/bs-$3.log || { echo "$3 FAILED"; grep -aE "Error|error:" logs/bs-$3.log | tail -3; exit 1; }
echo "== $3 (MAXREQ=$1 GRAPH_BS=$2) =="
