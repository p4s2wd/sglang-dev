#!/bin/bash
# Production launch: DeepSeek-V4-Flash on 8x RTX 2080 Ti (SM75).
# TP2+PP4 (NVLink pairs), 256K ctx, FP8 KV, decode CUDA graph, committed-default kernels.
# SGLANG_SM120_FLASHMLA_BACKEND=triton: fused sparse-MLA path, measured prefill +50%,
# decode +1.6~8.3% (runbook "当前最优配置"); the code default falls back to torch on SM75.
cd /data/nvme/sglang-codex
export MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2" CTX=262144
export SGLANG_SM120_FLASHMLA_BACKEND=triton
bash ./serve.sh 2 4 serve-prod --max-total-tokens 270000
