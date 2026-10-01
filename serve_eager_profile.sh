#!/bin/bash
# Eager profile server: no CUDA graph, no overlap schedule.
#
# Attributing the ~106 small decode kernels to the aten op that launches them
# needs the cpu_op -> cuda_runtime -> kernel correlation chain intact. Two things
# break it, and both are off here:
#   - CUDA graph replay: every kernel joins cudaGraphLaunch, so the op link is lost
#   - the overlap scheduler: the launching op is recorded on a different thread
#     than the kernel, so the correlation join misses (measured: 4260/4260 unlinked)
# The per-layer op mix is unchanged, so the attribution transfers to production.
set -u
cd /data/nvme/sglang-codex
. ./env.sh
mkdir -p logs
# SGLANG_SM120_FLASHMLA_BACKEND=triton swaps the pure-PyTorch sparse-decode
# fallback (81 small kernels/layer, per attrib_device_by_line.py) for the fused
# Triton kernel on the same call site. Set to torch to compare.
# This must stay above the exec: a # comment inside a backslash continuation
# terminates the command, which silently dropped every flag below it.
export SGLANG_SM120_FLASHMLA_BACKEND="${SGLANG_SM120_FLASHMLA_BACKEND:-triton}"
exec python -m sglang.launch_server \
  --model "$DSV4_CKPT" --tokenizer-path "$DSV4_CKPT" \
  --load-format dummy \
  --json-model-override-args '{"num_hidden_layers": 4}' \
  --tp-size 2 --pp-size 1 \
  --mem-fraction-static 0.62 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 4096 \
  --chunked-prefill-size 512 \
  --max-running-requests 1 \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --host 127.0.0.1 --port 30000 \
  2>&1 | tee logs/profile-eager3.log
