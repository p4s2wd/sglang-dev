#!/bin/bash
# usage: ab_maxreq2.sh <MAXREQ> <tag>
# Decisive test of whether MAXREQ=2 is actively harmful at the user's operating
# point. scheduler.py:1175 sets pp_max_micro_batch_size = max(max_running_requests
# // pp_size, 1); with MAXREQ=2 and PP=4 that is 1, so only one request may be
# admitted per batch no matter how many clients connect -- the server log confirms
# all 228 decode steps ran at #running-req: 1 even with two HTTP clients in flight.
#
# The user's constraint is "1-2 requests is enough". MAXREQ=8 raises the cap to
# max(8//4,1)=2, which lets exactly those two requests share a decode step. So this
# is not about exceeding the user's concurrency limit -- it is about whether the
# limit they asked for is actually honored by the scheduler.
#
# Measured with exactly 2 concurrent, equal-length, uuid-salted requests so they
# start and finish together, against the same probe for both settings.
cd /data/nvme/sglang-codex
MR=$1; TAG=$2
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 50
for i in $(seq 1 15); do L=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$L" = "0" ] && break; sleep 10; done
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=$MR GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 2 4 q-$TAG --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 70); do sleep 10; grep -qaE "fired up|Error|out of memory" logs/q-$TAG.log 2>/dev/null && break; done
grep -qaE "fired up" logs/q-$TAG.log || { echo "SERVER FAILED"; exit 1; }
echo "=== MAXREQ=$MR (admission cap = $((MR/4>1?MR/4:1))) ==="
for n in 1 2 1 2; do timeout 400 .venv/bin/python dec_overlap.py $n 64 logs/q-$TAG.log 2>&1 | head -1; done
