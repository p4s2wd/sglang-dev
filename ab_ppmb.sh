#!/bin/bash
# usage: ab_ppmb.sh <MAXREQ> <PP_MB> <tag>
# Fill the PP pipeline with one request's own chunks. pp_max_micro_batch_size
# defaults to max_running_requests // pp_size, which at MAXREQ=2/PP=4 is 1 -- one
# batch in flight, so stage 0 cannot start chunk i+1 while chunk i is in stage 1.
# That is the 75-point serialisation, and it is a config knob, not a hard limit.
cd /data/nvme/sglang-codex
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 45
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=$1 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton \
  bash ./serve.sh 2 4 ppmb-$3 --pp-max-micro-batch-size $2 --max-total-tokens 270000 \
  >/dev/null 2>&1 </dev/null &
for i in $(seq 1 60); do
  sleep 10
  grep -qaE "fired up" logs/ppmb-$3.log 2>/dev/null && break
done
grep -qaE "fired up" logs/ppmb-$3.log || { echo "$3 FAILED"; grep -aE "Error|error" logs/ppmb-$3.log | tail -3; exit 1; }
echo "== $3 (MAXREQ=$1 PP_MB=$2) =="
for t in 1 2 3; do
  timeout 400 .venv/bin/python probe_prefill_rate2.py 8000 "t$t" 2>&1 | tail -1
done
echo -n "batches with >1 new-seq: "
grep -acE "new-seq: [2-9]" logs/ppmb-$3.log 2>/dev/null || echo 0
