#!/bin/bash
# usage: ab_nt.sh <NT> <tag>
# Sweep SGLANG_SM75_W4A16_V3_NT end to end.
#
# The production DECODE trace shows w4a16_v3_kernel launched as grid [64,6,1] with
# 32-thread blocks: 384 warps spread over 68 SMs = 5.6 warps per SM, against a per-SM
# capacity of 32. Achieved occupancy 18-26%. A memory-bound kernel cannot reach peak
# bandwidth with that few loads in flight, which is exactly why production runs at
# 227 GB/s (37% of 616) rather than near peak.
#
# grid.x = ceil(n_tiles / NT) with n_tiles = n/8, so NT=4 gives 64 and NT=1 gives 256:
# four times the blocks and four times the warps in flight, at the cost of re-reading
# the activation per n tile (the reason commit fb2a2f7e70 chose NT=4). Whether the
# extra parallelism outweighs the extra activation traffic is a measurement, and the
# isolated benchmark is not trustworthy here -- it reported 137 GB/s where production
# measures 227, so its shapes do not match.
cd /data/nvme/sglang-codex
NT=$1; TAG=$2
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 50
for i in $(seq 1 15); do L=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$L" = "0" ] && break; sleep 10; done
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=$NT SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 2 4 nt-$TAG --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 70); do sleep 10; grep -qaE "fired up|out of memory|Traceback" logs/nt-$TAG.log 2>/dev/null && break; done
grep -qaE "fired up" logs/nt-$TAG.log || { echo "SERVER FAILED NT=$NT"; grep -aE "out of memory|Traceback" logs/nt-$TAG.log|tail -2|cut -c1-140; exit 1; }
echo "=== NT=$NT ==="
grep -aoE "max_total_num_tokens=[0-9]+" logs/nt-$TAG.log | tail -1
for r in 1 2 3; do timeout 400 bash probe_perf.sh 3 64 2>&1 | grep -a "decode throughput"; done
