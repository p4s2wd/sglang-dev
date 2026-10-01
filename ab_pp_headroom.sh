#!/usr/bin/env bash
# Keep the 8.8% from the 11,11,11,10 split, but fix the OOM.
#
# Measured:
#   * split 11,11,11,10 gives 1543.4 tok/s vs 1417.4 for the shipped 10,11,11,11
#     (+8.8%), reproduced twice (1417.4/1543.4 and 1418.7/1543.4).
#   * but with that split PP0 carries 11 layers instead of 10, and PP0's free
#     memory after decode-graph capture drops to 0.42 GB (PP1 and PP2, also 11
#     layers, keep 1.40/1.34 GB). Running a 200-question eval then died in
#     ncclAllGather with unhandled cuda error.
#   * PP0 must therefore stay at 10 layers. With 43 layers that forces
#     10,11,11,11, which is what ships.
#
# The remaining lever is memory headroom, not layers. max-total-tokens 270000 at
# mem-fraction 0.97 leaves almost nothing; shrinking the KV pool gives PP0 room
# while keeping the fast split. Cost: a smaller KV cache, which is a capacity
# trade-off, not a correctness one, so the arms below report it explicitly.
#
# Arms: fast split with progressively smaller pools, plus the shipped split at
# the same pool so the comparison is like-for-like.
#
# Usage: ab_pp_headroom.sh [rounds]
set -uo pipefail
ROUNDS="${1:-1}"
LENS="8000,13000"
LAUNCH=/data/nvme/sglang/deepseek-v4-flash.sh
cd /data/nvme/sglang

env_of() { tr '\0' '\n' < "/proc/$(pgrep -f 'sglang serve' | head -1)/environ" 2>/dev/null; }

stop_and_confirm() {
  timeout 200 "$LAUNCH" --stop >/dev/null 2>&1
  for _ in $(seq 1 24); do
    timeout 5 curl -s -m 4 http://127.0.0.1:8200/health -o /dev/null 2>/dev/null || return 0
    sleep 5
  done
  return 1
}

# run_arm <label> <partition|default> <max_total_tokens>
run_arm() {
  local label="$1" part="$2" mtt="$3"
  stop_and_confirm || { echo "== $label SKIPPED (old server up)"; return; }
  local envargs=()
  [ "$part" != "default" ] && envargs+=(SGLANG_PP_LAYER_PARTITION="$part")
  setsid nohup env "${envargs[@]}" MAX_TOTAL_TOKENS="$mtt" "$LAUNCH" \
      >/tmp/opencode/ph_restart.log 2>&1 </dev/null &
  disown
  local ok=1
  for _ in $(seq 1 40); do
    sleep 10
    if timeout 5 curl -s -m 4 http://127.0.0.1:8200/health -o /dev/null 2>/dev/null; then ok=0; break; fi
  done
  if [ $ok -ne 0 ]; then
    echo "== $label FAILED TO START =="
    grep -oE "CUDA out of memory|OutOfMemoryError" logs/serve-prod.log 2>/dev/null | sort -u | tail -1
    return
  fi
  # avail mem after decode graph capture is the number that decided OOM before
  local av
  av=$(grep -oE "Capture target decode CUDA graph end.*avail mem=[0-9.]+ GB" \
        logs/serve-prod.log 2>/dev/null | grep -oE "avail mem=[0-9.]+ GB" | sort -t= -k2 -n | head -1)
  local got
  got=$(env_of | sed -n 's/^SGLANG_PP_LAYER_PARTITION=//p')
  echo "== $label  (split=${got:-default} pool=$mtt  min avail mem: ${av:-?}) =="
  cd /data/nvme/sglang-codex
  timeout 1200 .venv/bin/python pf_probe_prod.py --port 8200 --lens "$LENS" \
      --rounds 1 --no-warmup 2>&1 | tail -3
  cd /data/nvme/sglang
}

for r in $(seq 1 "$ROUNDS"); do
  run_arm "r$r fast 11,11,11,10 pool230k" 11,11,11,10 230000
  run_arm "r$r ship 10,11,11,11 pool230k" default    230000
done
echo "AB_DONE"
