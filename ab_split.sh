#!/bin/bash
# usage: ab_split.sh
# A/B the head-shared topk split end to end at a context long enough to saturate topk.
#
# The isolated kernel test showed split=16 is 2.30x faster than the per-head kernel at
# bs=1, but it ran on an 8192-token pool whose L2 hit rate is far above production's
# 236800, and production decode is bs=1 with the gate at B>=2, so the split is currently
# unreachable in production. Both facts need an end-to-end number.
#
# Context choice: topk saturates once the c128 compressed cache exceeds index_topk=512,
# i.e. past ~65536 prompt tokens. 86K is the shortest context where every attention call
# gathers a full 512 tiles, and its prefill is ~115 s instead of the 811 s a 216K prompt
# costs. Measuring here captures the real attention shape at a fraction of the runtime.
#
# Method: warm the prompt into the radix cache, then price the decode window as the
# difference between two cache-warm calls (41 new tokens minus 1), which cancels cached
# prefill and queue overhead.
cd /data/nvme/sglang-codex
. ./env.sh

run_cfg () {
  local TAG=$1; shift
  pkill -9 -f "launch_serve[r]" 2>/dev/null
  sleep 50
  for i in $(seq 1 15); do L=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$L" = "0" ] && break; sleep 10; done
  setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
    SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 "$@" \
    bash ./serve.sh 2 4 ab-$TAG --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
  for i in $(seq 1 70); do sleep 10; grep -qaE "fired up|out of memory|Traceback" logs/ab-$TAG.log 2>/dev/null && break; done
  grep -qaE "fired up" logs/ab-$TAG.log || { echo "$TAG SERVER FAILED"; grep -aE "out of memory|Traceback" logs/ab-$TAG.log|tail -2|cut -c1-140; return 1; }
  echo "=== $TAG ==="
  timeout 1500 .venv/bin/python dec_ctx_ab.py 2>&1 | grep -aE "ctx|FAIL"
  timeout 400 .venv/bin/python bs8_correct.py 2>&1 | grep -a "bs=8 batched"
}

# A: production as it stands -- gate at 2 means bs=1 takes the per-head kernel
run_cfg baseline SGLANG_SM75_HS_TOPK_SPLIT=1
# B: gate lowered to 1 so bs=1 reaches head-shared, with the split on
run_cfg split16 SGLANG_SM75_HS_TOPK_SPLIT=16 SGLANG_SM75_HEADSHARED_MIN_BATCH=1
echo DONE
