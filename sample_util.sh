#!/bin/bash
# Sample GPU utilization while a generation runs, to tell a compute-bound
# server from one that is idle waiting on the CPU.
# Usage: ./sample_util.sh <seconds> [prompt_tokens]
SECS="${1:-20}"
OUT=/tmp/util.$$.csv

( for i in $(seq "$SECS"); do
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits 2>/dev/null | awk -F, '{s+=$2} END {print s/NR}'
    sleep 1
  done > "$OUT" ) &
SAMP=$!

bash /data/nvme/sglang-codex/probe_generate.sh \
  "Explain in detail how a steam turbine converts heat into rotational work, covering the Rankine cycle." 120 \
  >/dev/null 2>&1

wait $SAMP 2>/dev/null
echo "GPU util samples over ${SECS}s of generation (mean of 8 GPUs):"
sort -n "$OUT" | uniq -c | awk '{printf "  %5.1f%% x%s\n", $2, $1}' | tail -12
echo "--- summary ---"
awk '{n++; s+=$1; if($1<10) idle++} END {printf "  n=%d mean=%.1f%%  samples_below_10%%=%d (%.0f%%)\n", n, s/n, idle, 100*idle/n}' "$OUT"
rm -f "$OUT"
