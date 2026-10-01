#!/bin/bash
# usage: ab_v3.sh <NT> <tag>   -- NT=0 shipped kernel, NT=4 v3
# End-to-end A/B for the v3 expert kernel. Decode bs=1 and bs=8 plus a prefill
# probe, all against the same boot so thermal drift does not masquerade as a
# regression (the lesson from the false 656-vs-702 result).
cd /data/nvme/sglang-codex
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
NT=$1; TAG=$2
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=8 GRAPH_MAX_BS=8 \
  GRAPH_BS="1 2 4 8" SGLANG_SM75_FLASHMLA_BACKEND=triton \
  SGLANG_SM75_W4A16_V3_NT=$NT \
  bash ./serve.sh 2 4 v3-$TAG --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 60); do
  sleep 10
  grep -qaE "fired up|Error|Traceback" logs/v3-$TAG.log 2>/dev/null && break
done
grep -qaE "fired up" logs/v3-$TAG.log || { echo "SERVER FAILED"; grep -aE "Error|Traceback" logs/v3-$TAG.log | tail -3 | cut -c1-150; exit 1; }
echo "=== NT=$NT ($TAG) ==="
grep -aoE "max_total_num_tokens=[0-9]+" logs/v3-$TAG.log | tail -1
bash ./probe_perf.sh 3 64 2>&1 | grep -aE "decode throughput|prefill"
timeout 300 .venv/bin/python decode_bs.py 8 64 2>&1 | tail -1
timeout 300 .venv/bin/python decode_bs.py 1 64 2>&1 | tail -1
