#!/bin/bash
# Launch DeepSeek-V4-Flash on 8x RTX 2080 Ti (SM75).
# Topology: NVLink pairs are (0,1) (2,3) (4,5) (6,7); cross-pair is only PCIe.
# TP2 lands inside one NVLink pair, so TP2+PP4 is the comm-optimal split.
#
# Memory: measured weights occupy 96.83% of a 22 GiB card at TP2xPP4, so
# --mem-fraction-static must sit above 0.969 or the KV pool gets nothing.
# Activation headroom is therefore thin: keep the prefill chunk small.
#
# Usage: ./serve.sh <tp> <pp> <logname> [extra args...]
set -u
cd /data/nvme/sglang-codex
. ./env.sh

TP="${1:-2}"
PP="${2:-4}"
LOG="${3:-serve-tp${TP}pp${PP}}"
FRAC="${FRAC:-0.97}"
CHUNK="${CHUNK:-512}"
CTX="${CTX:-1024}"
PREFILL_GRAPH="${PREFILL_GRAPH:-disabled}"
PREFILL_BS="${PREFILL_BS:-512}"
PREFILL_MAX_BS="${PREFILL_MAX_BS:-512}"
shift 3 2>/dev/null || true

exec python -m sglang.launch_server \
  --model /data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731 \
  --tp-size "$TP" --pp-size "$PP" \
  --mem-fraction-static "$FRAC" \
  --kv-cache-dtype fp8_e4m3 \
  --context-length "$CTX" \
  --chunked-prefill-size "$CHUNK" \
  --max-running-requests "${MAXREQ:-1}" \
  --cuda-graph-backend-decode "${DECODE_GRAPH:-full}" \
  --cuda-graph-max-bs-decode "${GRAPH_MAX_BS:-1}" \
  --cuda-graph-bs-decode ${GRAPH_BS:-1} \
  --cuda-graph-backend-prefill "$PREFILL_GRAPH" \
  ${PREFILL_GRAPH:+--cuda-graph-bs-prefill "$PREFILL_BS"} \
  ${PREFILL_GRAPH:+--cuda-graph-max-bs-prefill "$PREFILL_MAX_BS"} \
  --host 127.0.0.1 --port 30000 \
  "$@" 2>&1 | tee "/data/nvme/sglang-codex/logs/${LOG}.log"
