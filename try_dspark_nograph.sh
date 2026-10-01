#!/bin/bash
# usage: try_dspark_nograph.sh <MAX_TOKENS> <tag>
# The OOM is not the KV pool: capping tokens at 60000 (35 MB of KV) still died at
# the same 256 MiB allocation with 243 MiB free, so the missing ~1 GB/card is
# held by CUDA graph capture and its memory pool, not by KV. DSpark's draft adds
# ~0.5 GB/card of markov-head weights on top of a target that already fills the
# card at TP8. Drop graph capture to free the space, measure decode WITHOUT
# graphs, and treat any DSpark win as needing graphs back later.
cd /data/nvme/sglang-codex
MT=$1; TAG=$2
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 8 1 dng-$TAG --max-total-tokens $MT --disable-cuda-graph \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 90); do
  sleep 10
  grep -qaE "fired up|Error|Traceback|out of memory" logs/dng-$TAG.log 2>/dev/null && break
done
if grep -qaE "fired up" logs/dng-$TAG.log; then
  echo "DSPARK PP1/TP8 NOGRAPH UP (max_total_tokens=$MT)"
  grep -aoE "max_total_num_tokens=[0-9]+" logs/dng-$TAG.log | tail -1
else
  echo "DSPARK PP1/TP8 NOGRAPH FAILED (max_total_tokens=$MT)"
  grep -anE "out of memory|Error|assert" logs/dng-$TAG.log | grep -av coredump | tail -3 | cut -c1-160
fi
