#!/bin/bash
# usage: try_dspark_pp1b.sh <MAX_TOKENS> <tag>
# FRAC turned out to be inert here: --max-total-tokens overrides the FRAC-derived
# pool size, which is why 0.955 and 0.88 both OOMed with the identical 243.25 MiB
# free. The cap is the knob. Sweep it down until DSpark starts, then measure
# decode throughput and the resulting context limit, so the DSpark decision rests
# on measured numbers rather than the earlier PP1/TP8 note (which predates the v3
# expert kernel and the transposed attention).
cd /data/nvme/sglang-codex
MT=$1; TAG=$2
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 8 1 dsm-$TAG --max-total-tokens $MT \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 90); do
  sleep 10
  grep -qaE "fired up|Error|Traceback|out of memory" logs/dsm-$TAG.log 2>/dev/null && break
done
if grep -qaE "fired up" logs/dsm-$TAG.log; then
  echo "DSPARK PP1/TP8 UP (max_total_tokens=$MT)"
  grep -aoE "max_total_num_tokens=[0-9]+" logs/dsm-$TAG.log | tail -1
  grep -aiE "dspark|markov" logs/dsm-$TAG.log | tail -3 | cut -c1-150
else
  echo "DSPARK PP1/TP8 FAILED (max_total_tokens=$MT)"
  grep -anE "out of memory|Error|assert" logs/dsm-$TAG.log | grep -av coredump | tail -3 | cut -c1-160
fi
