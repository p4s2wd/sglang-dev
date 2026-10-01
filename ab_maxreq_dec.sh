#!/bin/bash
# usage: ab_maxreq_dec.sh <MAXREQ> <tag>
# Test whether decode can batch at all. scheduler.py:1175 sets
#   pp_max_micro_batch_size = max(max_running_requests // pp_size, 1)
# and get_num_allocatable_reqs (scheduler.py:3627) returns
#   pp_max_micro_batch_size - running_bs,
# capping how many sequences enter one batch. Production runs MAXREQ=2 with PP=4, so
# that is max(2//4,1)=1: exactly one request may be admitted at a time, no matter how
# many clients connect. The server log confirms it -- all 228 decode batch lines say
# "#running-req: 1", even when two HTTP clients are in flight simultaneously.
#
# This matters because decode is 16 tok/s at bs=1 and 30 tok/s aggregate at bs=2, and
# the bandwidth floor for bs=2 is 125 tok/s. Batching is the single largest decode
# lever, and it is gated by an argument, not by hardware. MAXREQ=8 gives
# max(8//4,1)=2, which should allow bs=2.
cd /data/nvme/sglang-codex
MR=$1; TAG=$2
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 50
for i in $(seq 1 15); do L=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$L" = "0" ] && break; sleep 10; done
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=$MR GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 2 4 mr-$TAG --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 70); do sleep 10; grep -qaE "fired up|Error|out of memory" logs/mr-$TAG.log 2>/dev/null && break; done
grep -qaE "fired up" logs/mr-$TAG.log || { echo "SERVER FAILED"; grep -aE "Error|out of memory" logs/mr-$TAG.log|tail -2|cut -c1-140; exit 1; }
echo "=== MAXREQ=$MR (pp micro-batch = $((MR/4))) ==="
grep -aoE "max_total_num_tokens=[0-9]+" logs/mr-$TAG.log | tail -1
timeout 700 .venv/bin/python dec_bs2.py 2>&1 | tail -3
echo "--- observed batch sizes ---"
timeout 200 .venv/bin/python dec_bs_probe.py logs/mr-$TAG.log 2>&1 | tail -2
