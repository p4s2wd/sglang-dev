#!/bin/bash
# Profile the PRODUCTION server during one prefill: 43 layers, real weights,
# TP2/PP4, CUDA graph on. The 4-layer dummy server used for source-line
# attribution cannot answer "what does prefill actually spend on", because it
# drops pipeline parallelism and the real per-layer mix.
set -u
cd /data/nvme/sglang-codex
. ./env.sh
PY=/data/nvme/sglang-codex/.venv/bin/python
OUT=/data/nvme/sglang-codex/profiles/$(date +%Y%m%d-%H%M%S)-prod-prefill
mkdir -p "$OUT"

best_effort() { curl -s -m 10 "http://127.0.0.1:30000$1" >/dev/null 2>&1 || true; }

# no stack traces: with 43 layers x 100+ kernels the event count explodes
curl -s -m 20 -X POST http://127.0.0.1:30000/start_profile \
  -H 'Content-Type: application/json' \
  -d "{\"output_dir\": \"$OUT\", \"activities\": [\"CUDA\"], \"with_stack\": false, \"num_steps\": 1}" >/dev/null 2>&1

timeout 600 $PY probe_prefill_rate2.py 8000 "prodprof" 2>&1 | tail -1
sleep 2
best_effort /stop_profile
sleep 6
ls -la "$OUT" | tail -6
