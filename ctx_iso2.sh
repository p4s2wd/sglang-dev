#!/bin/bash
# Same isolation as ctx_iso.sh but the decode CUDA graph can be switched off.
# If the --context-length penalty disappears without the graph, the capture-time
# max_seq_len is the mechanism; if it persists, the penalty is elsewhere.
set -u
cd /data/nvme/sglang-codex
. ./env.sh
CTX="$1"; LOG="$2"; GRAPH="${3:-full}"
ARGS=()
[ "$GRAPH" = "disabled" ] && ARGS+=(--disable-cuda-graph)
exec python -m sglang.launch_server \
  --model "$DSV4_CKPT" --tokenizer-path "$DSV4_CKPT" \
  --load-format dummy \
  --json-model-override-args '{"num_hidden_layers": 4}' \
  --tp-size 2 --pp-size 1 \
  --mem-fraction-static 0.62 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length "$CTX" \
  --chunked-prefill-size 512 \
  --max-running-requests 2 \
  "${ARGS[@]}" \
  --host 127.0.0.1 --port 30000 \
  2>&1 | tee "logs/${LOG}.log"
