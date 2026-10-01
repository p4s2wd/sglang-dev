#!/bin/bash
# usage: ab_qktrans.sh <QK_TRANS> <tag>
# End-to-end prefill A/B for the transposed QK gather. Kernel-level 1.45x on
# 39.6% of prefill device time predicts ~+20% end to end. MAXREQ=2 config, the
# one prefill was characterized at; three salted probes per boot.
cd /data/nvme/sglang-codex
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 \
  SGLANG_SM75_HS_QK_TRANS=$1 \
  bash ./serve.sh 2 4 qt-$2 --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 60); do
  sleep 10
  grep -qaE "fired up|Error|Traceback" logs/qt-$2.log 2>/dev/null && break
done
grep -qaE "fired up" logs/qt-$2.log || { echo "SERVER FAILED"; grep -aE "Error|Traceback" logs/qt-$2.log | tail -3 | cut -c1-150; exit 1; }
echo "=== QK_TRANS=$1 ($2) ==="
for i in 1 2 3; do
  timeout 300 .venv/bin/python probe_prefill_rate2.py 14000 2>&1 | grep -aoE "[0-9.]+ tok/s" | tail -1
done
bash ./probe_perf.sh 3 64 2>&1 | grep -a "decode throughput"
