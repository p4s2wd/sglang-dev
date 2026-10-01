#!/bin/bash
# usage: ab_ppmicro.sh <MAXREQ> <MAXPRETOK-or-none> <tag>
# Test the real serialization mechanism. In scheduler.py:1175
#   pp_max_micro_batch_size = max(max_running_requests // pp_size, 1)
# and get_num_allocatable_reqs (scheduler.py:3627) returns
#   pp_max_micro_batch_size - running_bs,
# which caps how many sequences enter one prefill batch. With MAXREQ=2 and PP=4
# that is max(2//4,1)=1, so exactly one sequence per batch -- which is precisely
# what every log line shows ("#new-seq: 1"). The claim under test is that
# --max-prefill-tokens is the limiter; its default is already 16384, so that
# cannot explain a cap of 1. MAXREQ is the knob that follows from the code, so
# this sweeps MAXREQ (2 -> 8 -> 16) holding everything else fixed, and measures
# prefill interleaved. If the mechanism is right, MAXREQ>=8 (=> micro-batch 2)
# and MAXREQ>=16 (=> 4) should raise aggregate prefill throughput.
cd /data/nvme/sglang-codex
MR=$1; MP=$2; TAG=$3
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
EXTRA=""
[ "$MP" != "none" ] && EXTRA="--max-prefill-tokens $MP"
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=$MR GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 2 4 pm-$TAG --max-total-tokens 270000 $EXTRA >/dev/null 2>&1 </dev/null &
for i in $(seq 1 60); do
  sleep 10
  grep -qaE "fired up|Error|Traceback|out of memory" logs/pm-$TAG.log 2>/dev/null && break
done
grep -qaE "fired up" logs/pm-$TAG.log || { echo "SERVER FAILED MAXREQ=$MR"; grep -aE "Error|out of memory" logs/pm-$TAG.log|tail -2|cut -c1-140; exit 1; }
echo "=== MAXREQ=$MR maxprefilltok=$MP ($TAG) ==="
grep -aoE "max_total_num_tokens=[0-9]+" logs/pm-$TAG.log | tail -1
# 3 concurrent requests so the microbatch budget actually binds
timeout 900 .venv/bin/python pf_conc2.py 11000 1,3 2>&1 | grep -aE "^n=|BEST"
