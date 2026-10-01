#!/bin/bash
# usage: ab_ppshape2.sh <TP> <PP> <FRAC> <tag>
# Same as ab_ppshape.sh but with a tunable --mem-fraction-static, because TP8/PP1
# refuses to start at 0.955: with PP=1 every card holds all 43 layers, and the parts
# TP cannot shard (compressor, indexer, router, norms, lm_head) are replicated 8 times
# instead of 2, so weights need FRAC > 0.966 before any KV cache is left.
#
# The point of the sweep is the decode/context tradeoff. TP4/PP2 measured 19.14 tok/s
# with max_total_num_tokens=122880 against TP2/PP4's 16.2 at 236800: fewer pipeline
# stages means less bubble per token, but MLA replicates the KV cache across TP ranks,
# so raising TP buys decode by spending context. TP8/PP1 is the end of that curve.
cd /data/nvme/sglang-codex
TP=$1; PP=$2; FR=$3; TAG=$4
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 50
for i in $(seq 1 15); do L=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$L" = "0" ] && break; sleep 10; done
setsid nohup env CTX=262144 CHUNK=512 FRAC=$FR MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh $TP $PP p2-$TAG --max-total-tokens 270000 >/dev/null 2>&1 </dev/null &
for i in $(seq 1 90); do sleep 10; grep -qaE "fired up|out of memory|Traceback|leave no GPU" logs/p2-$TAG.log 2>/dev/null && break; done
grep -qaE "fired up" logs/p2-$TAG.log || { echo "SERVER FAILED TP=$TP PP=$PP FRAC=$FR"; grep -aE "leave no GPU|out of memory|Traceback" logs/p2-$TAG.log|tail -2|cut -c1-160; exit 1; }
echo "=== TP=$TP PP=$PP FRAC=$FR ==="
grep -aoE "max_total_num_tokens=[0-9]+" logs/p2-$TAG.log | tail -1
timeout 400 bash probe_perf.sh 5 64 2>&1 | grep -a "decode throughput"
timeout 400 .venv/bin/python dec_overlap.py 1 64 logs/p2-$TAG.log 2>&1 | head -1
