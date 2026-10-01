#!/bin/bash
# usage: try_dspark_off.sh <OFFLOAD_GB> <MAX_TOKENS> <tag>
# The DSpark gap is now measured, not guessed: the target finishes loading at
# 20.39 GB of a 21.49 GiB card leaving 0.78 GB, and the draft stage -- a full
# DeepSeek-V4 MoE layer (mtp.2 is 3.89 GB in the checkpoint, ~0.5 GB/card at
# TP8) plus the markov head (129280 x 256 x 2 bf16, x2) -- needs roughly 1.0
# GB/card. So it is short by ~0.2 GB/card, and no allocator trick or KV-cap
# change touches that because the OOM happens before the KV pool is allocated.
# --cpu-offload-gb is the lever that actually moves target weight bytes off the
# card. It will cost decode speed on the offloaded layers, so if DSpark starts,
# the decode number has to be read against that cost, not taken as a pure win.
cd /data/nvme/sglang-codex
GB=$1; MT=$2; TAG=$3
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 8 1 dso-$TAG --max-total-tokens $MT --disable-cuda-graph \
  --cpu-offload-gb $GB \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 100); do
  sleep 10
  grep -qaE "fired up|Error|Traceback|out of memory" logs/dso-$TAG.log 2>/dev/null && break
done
if grep -qaE "fired up" logs/dso-$TAG.log; then
  echo "DSPARK UP with cpu-offload=${GB}GB"
  grep -aoE "max_total_num_tokens=[0-9]+" logs/dso-$TAG.log | tail -1
else
  echo "DSPARK FAILED with cpu-offload=${GB}GB"
  grep -aE "out of memory|Error:" logs/dso-$TAG.log | grep -av coredump | tail -2 | cut -c1-170
fi
