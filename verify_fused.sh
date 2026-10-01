#!/bin/bash
# usage: verify_fused.sh
# End-to-end check of the fused merge at the two contexts that matter, with committed
# defaults and no environment overrides.
#
# The isolated numbers say the split+merge path went 0.1201 -> 0.0629 ms per call, which
# at 22 calls per step per stage is 1.26 ms per step per stage. Whether that survives the
# 4-stage pipeline and the nccl spin is the question; the previous end-to-end gain of the
# split over the per-head baseline was 11.00 -> 13.55 tok/s at 86K.
cd /data/nvme/sglang-codex
. ./env.sh
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 50
for i in $(seq 1 15); do L=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$L" = "0" ] && break; sleep 10; done
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 2 4 fused --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 70); do sleep 10; grep -qaE "fired up|out of memory|Traceback" logs/fused.log 2>/dev/null && break; done
grep -qaE "fired up" logs/fused.log || { echo "SERVER FAILED"; grep -aE "out of memory|Traceback" logs/fused.log|tail -2|cut -c1-140; exit 1; }
grep -aoE "max_total_num_tokens=[0-9]+" logs/fused.log | tail -1
echo "=== decode @86K (was 13.55 pre-fusion, per-head baseline 11.00) ==="
timeout 1500 .venv/bin/python dec_ctx_ab.py 2>&1 | grep -aE "ctx|FAIL"
echo "=== short context ==="
for r in 1 2 3; do timeout 400 bash probe_perf.sh 4 64 2>&1 | grep -a "decode throughput"; done
echo "=== correctness ==="
timeout 400 .venv/bin/python bs8_correct.py 2>&1 | grep -a "bs=8 batched"
echo "=== decode @216K (was 13.26) ==="
timeout 2400 .venv/bin/python dec_ctx3.py 150000 2>&1 | tail -2
echo DONE
