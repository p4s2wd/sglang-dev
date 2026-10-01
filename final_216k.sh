#!/bin/bash
# usage: final_216k.sh
# Measure the committed defaults at the objective's actual target: 256K-class context.
#
# 86K already saturates index_topk=512, so the attention shape is the same, but the
# report's headline number was taken at 216K and the objective says 256K, so the target
# operating point needs its own number. Also re-baseline prefill at 150 W, since every
# prefill figure in the report is from 190 W and the SM clock is what drops under a
# power cap -- prefill is compute-bound, so unlike the bandwidth-bound expert kernels it
# should move.
#
# Runs detached: a 216K uncached prefill costs ~800 s at these rates.
cd /data/nvme/sglang-codex
. ./env.sh
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 50
for i in $(seq 1 15); do L=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$L" = "0" ] && break; sleep 10; done
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 2 4 final150 --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 70); do sleep 10; grep -qaE "fired up|out of memory|Traceback" logs/final150.log 2>/dev/null && break; done
grep -qaE "fired up" logs/final150.log || { echo "SERVER FAILED"; grep -aE "out of memory|Traceback" logs/final150.log|tail -2|cut -c1-140; exit 1; }
grep -aoE "max_total_num_tokens=[0-9]+" logs/final150.log | tail -1
echo "=== prefill at 150 W ==="
timeout 900 .venv/bin/python pf_curve.py 7600,11500,23000 3 2>&1 | tail -5
echo "=== decode at 216K ==="
timeout 2400 .venv/bin/python dec_ctx3.py 150000 2>&1 | tail -2
echo "=== correctness ==="
timeout 400 .venv/bin/python bs8_correct.py 2>&1 | grep -a "bs=8 batched"
echo DONE
