#!/bin/bash
# usage: try_dspark.sh <FRAC> <tag> [extra args...]
# DSpark is the checkpoint's own self-drafting path: config.json carries
# dspark_block_size=5, dspark_markov_rank=256, dspark_target_layer_ids=[40,41,42]
# and the index has mtp.2.markov_head.markov_w1/w2, so the draft is bundled and
# --speculative-algorithm DSPARK alone enables it (no separate draft path). The
# earlier note that this was blocked on ~0.5 GiB/card assumed a full draft layer;
# a markov head is far smaller, so the real blocker is unknown until tried.
# FRAC is a knob because the KV pool is the thing that has to give.
cd /data/nvme/sglang-codex
FRAC=$1; TAG=$2; shift 2
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=$FRAC MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 2 4 ds-$TAG --max-total-tokens 270000 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 "$@" >/dev/null 2>&1 </dev/null &
for i in $(seq 1 90); do
  sleep 10
  grep -qaE "fired up|Error|Traceback|error" logs/ds-$TAG.log 2>/dev/null && break
done
if grep -qaE "fired up" logs/ds-$TAG.log; then
  echo "DSPARK SERVER UP (FRAC=$FRAC)"
  grep -aoE "max_total_num_tokens=[0-9]+" logs/ds-$TAG.log | tail -1
  grep -aiE "dspark|speculative|accept" logs/ds-$TAG.log | tail -6 | cut -c1-150
else
  echo "DSPARK SERVER FAILED (FRAC=$FRAC)"
  grep -anE "Error|error|assert|Traceback|out of memory" logs/ds-$TAG.log | grep -av coredump | tail -8 | cut -c1-170
fi
