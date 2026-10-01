#!/bin/bash
# usage: ab_hsmin.sh <MIN_BATCH> <tag>
# A/B the headshared gate. flash_mla_sm120_triton.py:408 sets
# _HEADSHARED_MIN_BATCH=16, and its docstring justifies it with "at B=2 it measured
# 0.7x (0.88ms vs 0.62ms)". That measurement predates fcb22c9d9d, which gathered the
# KV transposed so the QK dot needs no tl.trans and made the headshared kernel 1.45x
# faster. Re-measured at production cache size (236800 tokens = 138 MB), the crossover
# has moved to between bs=1 and bs=2:
#   bs   per-head  headshared  ratio
#    1     0.2883      0.5139   0.56x   (per-head wins)
#    2     0.5568      0.5169   1.08x
#    4     1.0938      0.5104   2.14x
#    8     2.1844      0.5209   4.19x
#   16     4.3253      0.6460   6.70x
# The headshared kernel is flat because MLA shares one KV entry across all 64 heads,
# so it amortizes the gather over 16 heads; the per-head kernel re-gathers it 64 times.
# So every batch from 2 to 15 is currently served by the slower kernel. This measures
# end-to-end decode at the operating point (bs=2) with the gate at 16 vs 2.
cd /data/nvme/sglang-codex
MB=$1; TAG=$2
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 50
for i in $(seq 1 15); do L=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$L" = "0" ] && break; sleep 10; done
echo "launching HEADSHARED_MIN_BATCH=$MB"
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  SGLANG_SM75_HEADSHARED_MIN_BATCH=$MB \
  bash ./serve.sh 2 4 hs-$TAG --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 70); do sleep 10; grep -qaE "fired up|Error|out of memory" logs/hs-$TAG.log 2>/dev/null && break; done
grep -qaE "fired up" logs/hs-$TAG.log || { echo "SERVER FAILED"; grep -aE "Error|out of memory" logs/hs-$TAG.log|tail -2|cut -c1-140; exit 1; }
grep -aoE "max_total_num_tokens=[0-9]+" logs/hs-$TAG.log | tail -1
echo "--- bs=1 ---"
timeout 400 bash probe_perf.sh 5 64 2>&1 | grep -a "decode throughput"
echo "--- bs=2 aggregate ---"
timeout 700 .venv/bin/python dec_bs2.py 2>&1 | tail -3
