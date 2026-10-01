#!/bin/bash
# Fast forward-path smoke: random weights, so a crash surfaces in ~1 minute
# instead of after a 10-minute weight load. Same construction and forward path
# as a real run -- only the disk read is skipped. Output values are garbage.
#
# Usage: ./serve_dummy.sh <tp> <pp> <logname> [extra args...]
set -u
cd /data/nvme/sglang-codex
. ./env.sh

TP="${1:-2}"
PP="${2:-4}"
LOG="${3:-dummy-tp${TP}pp${PP}}"
shift 3 2>/dev/null || true

exec python -m sglang.launch_server \
  --model /data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731 \
  --load-format dummy \
  --tp-size "$TP" --pp-size "$PP" \
  --mem-fraction-static 0.97 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length "${CTX:-1024}" \
  --chunked-prefill-size 512 \
  --max-running-requests "${MAXREQ:-1}" \
  --cuda-graph-backend-decode "${DECODE_GRAPH:-full}" \
  --cuda-graph-max-bs-decode "${GRAPH_MAX_BS:-1}" \
  --cuda-graph-bs-decode ${GRAPH_BS:-1} \
  --cuda-graph-backend-prefill "${PREFILL_GRAPH:-disabled}" \
  ${PREFILL_BS:+--cuda-graph-bs-prefill "$PREFILL_BS"} \
  ${PREFILL_MAX_BS:+--cuda-graph-max-bs-prefill "$PREFILL_MAX_BS"} \
  --host 127.0.0.1 --port 30000 \
  "$@" 2>&1 | tee "/data/nvme/sglang-codex/logs/${LOG}.log"
