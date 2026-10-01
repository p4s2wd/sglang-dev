#!/bin/bash
# usage: ab_depth.sh <DEPTH> <tag>
# Test --pp-async-batch-depth, the one knob that targets the measured prefill
# bottleneck. Each stage's proxy send -- the thing that unblocks the next stage --
# is sequenced AFTER that stage's blocking output recv and its d2h sync +
# process_batch_result (scheduler_pp_mixin.py:135-162), so every stage adds its CPU
# post-processing tail to the downstream critical path. At depth=0 the ring has zero
# slack (pp_loop_size == pp_size == 4). depth>0 both adds slack and moves the output
# recv before the launch.
#
# Prediction if the mechanism is right: per-stage cadence falls from the measured
# 538 ms toward the 150-172 ms of real device work per chunk.
#
# The previous attempt at this test died with OOM during weight loading because a
# leftover process held 3.54 GiB -- a harness artifact. So this script refuses to
# launch until the GPUs are verifiably empty.
cd /data/nvme/sglang-codex
D=$1; TAG=$2
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 50
for i in $(seq 1 20); do
  LEFT=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$LEFT" = "0" ] && break
  echo "waiting for $LEFT leftover compute process(es)"; sleep 10
done
LEFT=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$LEFT" != "0" ]; then echo "ABORT: $LEFT compute processes still hold the GPUs"; exit 1; fi
echo "GPUs empty; launching depth=$D"
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 2 4 dep-$TAG --max-total-tokens 270000 --pp-async-batch-depth $D \
  >/dev/null 2>&1 </dev/null &
for i in $(seq 1 70); do
  sleep 10
  grep -qaE "fired up|Error|Traceback|out of memory" logs/dep-$TAG.log 2>/dev/null && break
done
grep -qaE "fired up" logs/dep-$TAG.log || { echo "SERVER FAILED depth=$D"; grep -aE "Error|out of memory|raise" logs/dep-$TAG.log|tail -3|cut -c1-150; exit 1; }
echo "=== depth=$D ($TAG) ==="
grep -aoE "max_total_num_tokens=[0-9]+" logs/dep-$TAG.log | tail -1
timeout 1200 .venv/bin/python pf_curve.py 8000,12000 3 2>&1 | tail -8
