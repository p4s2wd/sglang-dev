#!/bin/bash
# usage: try_dspark_f.sh <OFFLOAD_GB> <FRAC> <MAX_TOKENS> <tag>
# DSpark's draft costs 1.46 GB/card (three TP-split stages) and the target at TP8
# leaves only 0.78 GB, so the gap is ~0.7 GB/card. Offloading 3 GB closes it but
# costs far more than the draft saves (2.98 tok/s). At 1 GB the loader complained
# that no room was left for KV and asked for --mem-fraction-static above 0.980 --
# FRAC is the ceiling on weights+KV together, so raising it is what lets a small
# offload leave a usable KV pool. Sweep the offload down and FRAC up to find the
# least offload that boots, because every offloaded byte is read over PCIe on
# every decode step.
cd /data/nvme/sglang-codex
GB=$1; FRAC=$2; MT=$3; TAG=$4
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=$FRAC MAXREQ=2 \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 8 1 dsf-$TAG --max-total-tokens $MT --disable-cuda-graph \
  --cpu-offload-gb $GB \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 100); do
  sleep 10
  grep -qaE "fired up|Error|Traceback|out of memory|no GPU memory" logs/dsf-$TAG.log 2>/dev/null && break
done
if grep -qaE "fired up" logs/dsf-$TAG.log; then
  echo "DSPARK UP offload=${GB}GB FRAC=$FRAC"
  grep -aoE "max_total_num_tokens=[0-9]+" logs/dsf-$TAG.log | tail -1
else
  echo "DSPARK FAILED offload=${GB}GB FRAC=$FRAC"
  grep -aE "out of memory|no GPU memory|Error:" logs/dsf-$TAG.log | grep -av coredump | tail -1 | cut -c1-150
fi
