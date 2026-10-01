#!/bin/bash
# usage: try_dspark_pp1.sh <FRAC> <tag>
# DSpark is hard-gated to pp_size==1 (speculative_hook.py:551). On this box
# PP=1/TP8 previously ran at 19.44 tok/s with the KV pool collapsing to 36352
# tokens, which is why it was rejected for the 256K goal. But that measurement
# predates the v3 expert kernel and the transposed attention, and the user's
# claim is a 1.5-3x decode speedup, so the trade deserves fresh numbers rather
# than the old verdict: measure decode throughput AND the real context limit
# with DSpark on, then decide.
cd /data/nvme/sglang-codex
FRAC=$1; TAG=$2
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=$FRAC MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 8 1 dsp-$TAG --max-total-tokens 270000 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 90); do
  sleep 10
  grep -qaE "fired up|Error|Traceback|out of memory" logs/dsp-$TAG.log 2>/dev/null && break
done
if grep -qaE "fired up" logs/dsp-$TAG.log; then
  echo "DSPARK PP1/TP8 UP (FRAC=$FRAC)"
  grep -aoE "max_total_num_tokens=[0-9]+" logs/dsp-$TAG.log | tail -1
  grep -aiE "dspark|markov|draft" logs/dsp-$TAG.log | tail -4 | cut -c1-150
else
  echo "DSPARK PP1/TP8 FAILED (FRAC=$FRAC)"
  grep -anE "Error|error|assert|Traceback|out of memory" logs/dsp-$TAG.log | grep -av coredump | tail -8 | cut -c1-170
fi
