#!/bin/bash
# usage: ab_radix.sh <on|off> <tag>
# Test whether per-chunk scheduler work is the remaining prefill cost.
#
# The PP-3 trace has two stalls (500 ms, 1506 ms) inside one request's prefill.
# During the 1506 ms one, PP2 ran 2066 kernels while PP0 and PP3 ran ZERO -- a
# first stage that launches nothing starves everyone downstream, and 1.5 s with no
# kernels on any rank is CPU work on the scheduler thread, not GPU work.
#
# The candidate: every chunk of a chunked request re-runs stash_chunked_request ->
# cache_unfinished_req -> radix insert + match_prefix over the WHOLE prompt so far
# (scheduler.py:3510-3511, radix_cache.py:538-598), plus init_next_round_input ->
# _refresh_fill_ids + full-length match_prefix (scheduler.py:3771,
# schedule_batch.py:1438-1470). That is O(prompt) per chunk, O(n^2) per request, and
# it is exactly the kind of work that shows up as an all-ranks GPU hole.
#
# --disable-radix-cache removes it. Cost: no prefix reuse, which this benchmark
# defeats with a uuid anyway, so the benchmark is unaffected while real repeated
# prefixes would lose the hit. If prefill rises materially, the O(n^2) is real and
# worth fixing properly instead of by disabling the cache.
cd /data/nvme/sglang-codex
M=$1; TAG=$2
pkill -9 -f "launch_serve[r]" 2>/dev/null
sleep 50
for i in $(seq 1 20); do
  LEFT=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$LEFT" = "0" ] && break
  echo "waiting for $LEFT leftover process(es)"; sleep 10
done
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != "0" ] && { echo ABORT; exit 1; }
EXTRA=""; [ "$M" = "off" ] && EXTRA="--disable-radix-cache"
echo "launching radix=$M"
setsid nohup env CTX=262144 CHUNK=512 FRAC=0.955 MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" \
  SGLANG_SM120_FLASHMLA_BACKEND=triton SGLANG_SM75_W4A16_V3_NT=4 SGLANG_SM75_HS_QK_TRANS=1 \
  bash ./serve.sh 2 4 rx-$TAG --max-total-tokens 270000 $EXTRA >/dev/null 2>&1 </dev/null &
for i in $(seq 1 70); do
  sleep 10
  grep -qaE "fired up|Error|Traceback|out of memory" logs/rx-$TAG.log 2>/dev/null && break
done
grep -qaE "fired up" logs/rx-$TAG.log || { echo "SERVER FAILED"; grep -aE "Error|out of memory|raise" logs/rx-$TAG.log|tail -3|cut -c1-150; exit 1; }
grep -aoE "max_total_num_tokens=[0-9]+" logs/rx-$TAG.log | tail -1
timeout 1400 .venv/bin/python pf_curve.py 8000,12000,24000 3 2>&1 | tail -6
