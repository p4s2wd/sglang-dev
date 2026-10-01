#!/bin/bash
# usage: ab_ppshape.sh <TP> <PP> <tag>
# Test whether fewer pipeline stages buy decode throughput.
#
# Measured per-stage DECODE device work per token (profiles/exact-1790058077, one
# request for 32 tokens at bs=1, so steps are known exactly):
#   PP0 23.23 ms  of which nccl SendRecv 15.51  -> compute  7.72
#   PP1 13.74 ms  of which nccl SendRecv  4.26  -> compute  9.48
#   PP2 13.95 ms  of which nccl SendRecv  5.49  -> compute  8.46
#   PP3 16.23 ms  of which nccl SendRecv  6.57  -> compute  9.66
# Sum of compute = 35.3 ms/token, but the measured wall step at bs=1 is 63.7 ms. The
# SendRecv time is a spin, not work: at bs=1 there is no other token to overlap, so a
# stage finishes and waits for the pipeline to drain. So ~28 ms of every decode step is
# pipeline bubble, and it is structural to PP=4 at batch 1 -- not a kernel problem.
#
# Any TP x PP = 8 puts 1/8 of the weights (19.5 GB) on each 21.5 GB card, so PP=2/TP4
# and PP=1/TP8 are memory-feasible and halve or remove the bubble. The cost is that
# NVLink pairs are (0,1)(2,3)(4,5)(6,7), so TP4 and TP8 allreduce cross the pair
# boundary. Measure both decode throughput and the resulting context capacity.
cd /data/nvme/sglang-codex
TP=$1; PP=$2; TAG=$3
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 50
for i in $(seq 1 15); do L=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$L" = "0" ] && break; sleep 10; done
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh $TP $PP pp-$TAG --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 70); do sleep 10; grep -qaE "fired up|Error|out of memory" logs/pp-$TAG.log 2>/dev/null && break; done
grep -qaE "fired up" logs/pp-$TAG.log || { echo "SERVER FAILED TP=$TP PP=$PP"; grep -aE "Error|out of memory" logs/pp-$TAG.log|tail -3|cut -c1-150; exit 1; }
echo "=== TP=$TP PP=$PP ==="
grep -aoE "max_total_num_tokens=[0-9]+" logs/pp-$TAG.log | tail -1
timeout 400 bash probe_perf.sh 5 64 2>&1 | grep -a "decode throughput"
timeout 400 .venv/bin/python dec_overlap.py 1 64 logs/pp-$TAG.log 2>&1 | head -1
