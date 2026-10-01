#!/bin/bash
# Small dummy-weight server for profiling.
#
# The real model leaves ~200 MiB free per card, and the torch sparse-MLA fallback
# materializes (chunk x topk x head_dim) fp32 intermediates -- a prefill chunk of
# 512 against index_topk 512 is ~500 MB, which OOMs before the profiler even gets
# a chance. Dropping to a few layers keeps the per-layer op mix (which is what a
# trace is for: the same kernels run per layer) while freeing enough memory to
# record GPU activity.
#
# Usage: ./serve_profile.sh [layers] [logname]
set -u
cd /data/nvme/sglang-codex
. ./env.sh

LAYERS="${1:-4}"
NAME="${2:-profile-srv}"
TP=2
# PP=1 on purpose: with one request in flight a pipeline is fully serialized, so
# PP bubbles would dominate the decode trace and hide the per-kernel breakdown
# this server exists to measure.
PP=1

# The torch sparse-MLA fallback materializes (chunk x topk x head_dim) fp32
# intermediates -- 512 x 512 x 512 x 4B is ~536 MB per prefill chunk. With dummy
# weights the KV pool would otherwise claim nearly all free memory (it sized
# itself at 19.8M tokens on the first try) and that gather OOMs. Cap the pool via
# --mem-fraction-static so the headroom is there; the chunk size stays at the
# production value so the trace reflects real prefill batch shapes.
FRAC="${FRAC:-0.4}"
CHUNK="${CHUNK:-512}"

mkdir -p logs
exec python -m sglang.launch_server \
  --model "$DSV4_CKPT" \
  --tokenizer-path "$DSV4_CKPT" \
  --load-format dummy \
  --json-model-override-args "{\"num_hidden_layers\": $LAYERS}" \
  --tp-size "$TP" --pp-size "$PP" \
  --mem-fraction-static "$FRAC" \
  --kv-cache-dtype fp8_e4m3 \
  --context-length "${CTX:-4096}" \
  --chunked-prefill-size "$CHUNK" \
  --max-running-requests "${MAXREQ:-1}" \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode "${GRAPH_MAX_BS:-1}" \
  --cuda-graph-bs-decode ${GRAPH_BS:-1} \
  --cuda-graph-backend-prefill "${PREFILL_GRAPH:-disabled}" \
  ${PREFILL_BS:+--cuda-graph-bs-prefill "$PREFILL_BS"} \
  ${PREFILL_MAX_BS:+--cuda-graph-max-bs-prefill "$PREFILL_MAX_BS"} \
  --host 127.0.0.1 --port 30000 \
  2>&1 | tee "logs/$NAME.log"
