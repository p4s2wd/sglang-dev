#!/bin/bash
# Isolate the --context-length decode penalty from everything else:
# dummy weights, 4 layers, TP2/PP1, CUDA graph ON (the capture path is the
# suspected mechanism). Only --context-length differs between runs.
set -u
cd /data/nvme/sglang-codex
. ./env.sh
CTX="$1"; LOG="$2"
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
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 2 --cuda-graph-bs-decode 1 2 \
  --host 127.0.0.1 --port 30000 \
  2>&1 | tee "logs/${LOG}.log"
