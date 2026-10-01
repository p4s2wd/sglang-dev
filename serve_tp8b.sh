#!/bin/bash
# usage: serve_tp8b.sh <CTX> <MAXTOK> <FRAC> <tag>
cd /data/nvme/sglang-codex
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=$1 CHUNK=512 FRAC=$3 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton \
  bash ./serve.sh 8 1 tp8-$4 --max-total-tokens $2 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 90); do
  sleep 10
  grep -qaE "fired up|Error|error:|Traceback" logs/tp8-$4.log 2>/dev/null && break
done
if grep -qaE "fired up" logs/tp8-$4.log; then
  echo "TP8/PP1 UP ctx=$1 maxtok=$2 frac=$3"
  grep -aoE "max_total_num_tokens=[0-9]+" logs/tp8-$4.log | tail -1
else
  echo "TP8/PP1 FAILED ctx=$1 maxtok=$2 frac=$3"
  grep -aE "ValueError|not enough|exceeds" logs/tp8-$4.log | tail -2 | cut -c1-170
fi
