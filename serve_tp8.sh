#!/bin/bash
# usage: serve_tp8.sh <CTX> <MAXTOK> <tag>
# Decode is PP4-serialised: one token traverses 4 stages in series, so its latency
# is 4 x 17.13 ms = 68.5 ms = 14.6 tok/s, matching the measured 14.9. The same 8
# GPUs arranged as PP1/TP8 put every layer on every card, so a token costs one
# stage's work instead of four -- roughly 4x lower latency. The blocker is memory:
# with all layers resident, the KV pool shrinks. This finds the largest context
# TP8/PP1 can actually serve.
cd /data/nvme/sglang-codex
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
CTX=$1; MAXTOK=$2; TAG=$3
setsid nohup env CTX=$CTX CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton \
  bash ./serve.sh 8 1 tp8-$TAG --max-total-tokens $MAXTOK >/dev/null 2>&1 </dev/null &
for i in $(seq 1 90); do
  sleep 10
  grep -qaE "fired up|Error|error:|Traceback" logs/tp8-$TAG.log 2>/dev/null && break
done
if grep -qaE "fired up" logs/tp8-$TAG.log; then
  echo "TP8/PP1 UP ctx=$CTX maxtok=$MAXTOK"
  grep -aoE "max_total_num_tokens=[0-9]+" logs/tp8-$TAG.log | tail -1
else
  echo "TP8/PP1 FAILED ctx=$CTX maxtok=$MAXTOK"
  grep -aE "ValueError|Error|not enough|exceeds" logs/tp8-$TAG.log | tail -3 | cut -c1-160
fi
