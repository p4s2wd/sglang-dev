#!/bin/bash
# usage: try_dspark_seg.sh <MAX_TOKENS> <tag> [extra env...]
# The DSpark OOM is marginal: 243 MiB free against a 256 MiB expert-weight
# allocation, per card, during draft construction. Two candidate fixes for a
# gap that small: expandable_segments (defragments the caching allocator so a
# contiguous 256 MiB can be assembled from free-but-split pages) and dropping
# the KV cap, which is irrelevant here anyway since the OOM precedes KV
# allocation. If it still fails, the gap is real weight bytes and no allocator
# trick will close it.
cd /data/nvme/sglang-codex
MT=$1; TAG=$2
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 8 1 dsg-$TAG --max-total-tokens $MT --disable-cuda-graph \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 90); do
  sleep 10
  grep -qaE "fired up|Error|Traceback|out of memory" logs/dsg-$TAG.log 2>/dev/null && break
done
if grep -qaE "fired up" logs/dsg-$TAG.log; then
  echo "DSPARK UP (max_total_tokens=$MT)"
  grep -aoE "max_total_num_tokens=[0-9]+" logs/dsg-$TAG.log | tail -1
else
  echo "DSPARK FAILED (max_total_tokens=$MT)"
  grep -aE "out of memory" logs/dsg-$TAG.log | tail -1 | cut -c1-200
fi
