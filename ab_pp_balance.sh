#!/usr/bin/env bash
# Load-balance the PP layer split. No code change: sglang reads
# SGLANG_PP_LAYER_PARTITION ("10,11,11,11") and get_pp_indices() uses it verbatim.
#
# Why this is worth testing. Measured per-stage compute for the shipped split
# 10/11/11/11 (43 layers), same profile, same measurement window:
#   PP0 215.2 ms / 10 layers = 21.5 ms per layer
#   PP1 254.8 / 11 = 23.2      PP2 247.4 / 11 = 22.5      PP3 274.6 / 11 = 25.0
# PP3 is the bottleneck and its per-layer cost is 16% above PP0's. PP3 also
# carries lm_head (129280 x 4096 fp16 = 1010 MiB) and the final norm, which is
# consistent with its 21915 MiB vs PP0's 20435 MiB.
#
# 43 does not divide by 4, so only integer splits are possible. The candidates
# that keep the sum at 43:
#   A 10,11,11,11 (shipped)  B 11,11,11,10  C 11,10,11,11
# B and C move the light stage to the END, which is the opposite of what the
# per-layer numbers suggest, so they are included as a control: if the gain is
# real, it should show up as a specific prediction, not a generic wobble.
#
# The read-back of the env off the running process matters: a failed start would
# otherwise silently measure the previous arm.
#
# Usage: ab_pp_balance.sh [rounds]
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

run_arm() {
  local label="$1" part="$2"
  stop_and_confirm || { echo "== $label SKIPPED (old server up)"; return; }
  if [ "$part" = "default" ]; then
    setsid nohup "$LAUNCH" >/tmp/opencode/pp_restart.log 2>&1 </dev/null &
  else
    setsid nohup env SGLANG_PP_LAYER_PARTITION="$part" "$LAUNCH" \
        >/tmp/opencode/pp_restart.log 2>&1 </dev/null &
  fi
  disown
  local ok=1
  for _ in $(seq 1 40); do
    sleep 10
    if timeout 5 curl -s -m 4 http://127.0.0.1:8200/health -o /dev/null 2>/dev/null; then ok=0; break; fi
  done
  if [ $ok -ne 0 ]; then
    echo "== $label FAILED TO START =="
    grep -oE "CUDA out of memory|OutOfMemoryError|Invalid partition[^\"]*|does not match[^\"]*" \
      logs/serve-prod.log 2>/dev/null | sort -u | tail -2
    return
  fi
  local got
  got=$(env_of | sed -n 's/^SGLANG_PP_LAYER_PARTITION=//p')
  echo "== $label  (server has: ${got:-<default>}) =="
  cd /data/nvme/sglang-codex
  timeout 1200 .venv/bin/python pf_probe_prod.py --port 8200 --lens "$LENS" \
      --rounds 1 --no-warmup 2>&1 | tail -3
  cd /data/nvme/sglang
}

for r in $(seq 1 "$ROUNDS"); do
  run_arm "r$r A shipped 10,11,11,11" default
  run_arm "r$r B 11,11,11,10"         11,11,11,10
  run_arm "r$r C 11,10,11,11"         11,10,11,11
  run_arm "r$r D 12,11,10,10"         12,11,10,10
done
echo "AB_DONE"
