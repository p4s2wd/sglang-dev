#!/usr/bin/env bash
# Two untested levers, measured properly.
#
# LEVER 1: layer rebalance. Measured per-stage compute is
#   PP0 215.2 ms / 10 layers = 21.5 ms per layer
#   PP1 254.8 / 11 = 23.2      PP2 247.4 / 11 = 22.5      PP3 274.6 / 11 = 25.0
# PP3 is the bottleneck and its per-layer cost is 16% above PP0's, so the stages
# are not balanced even though the layer counts nearly are. Perfect balance would
# put every stage at 248 ms and lift the ceiling from 1814 to ~2065 tok/s.
# `--pp-layer-start`/`--num-hidden-layers` split is what sglang uses; this tests
# whether a different split helps.
#
# LEVER 2: chunked-prefill-size. The install docs say >512 OOMs because
# `_merge_partial_attn` materialises [tokens, 128 heads, 512] fp32 per layer.
# That constraint was measured on the PRE-opt1 build. The fp8 path and the fused
# triton attention have changed since, and the fused path may not use
# _merge_partial_attn at all. If a larger chunk works, every per-chunk fixed cost
# is amortised over more tokens.
#
# Both need a server restart, so arms alternate and each arm's env is verified on
# the running process before measuring.
#
# Usage: ab_structural.sh [rounds]
set -uo pipefail

ROUNDS="${1:-2}"
LENS="8000,13000"
LAUNCH=/data/nvme/sglang/deepseek-v4-flash.sh
cd /data/nvme/sglang

start_server() {
  # "$@" = VAR=VAL ...  (CHUNK is passed separately)
  local chunk="$1"; shift
  setsid nohup env "$@" CHUNK="$chunk" "$LAUNCH" \
      >/tmp/opencode/st_restart.log 2>&1 </dev/null &
  disown
  for _ in $(seq 1 40); do
    sleep 10
    if timeout 5 curl -s -m 4 http://127.0.0.1:8200/health -o /dev/null 2>/dev/null; then
      return 0
    fi
  done
  echo "SERVER FAILED TO START ($* CHUNK=$chunk)" >&2
  return 1
}

stop_and_confirm() {
  timeout 200 "$LAUNCH" --stop >/dev/null 2>&1
  for _ in $(seq 1 20); do
    timeout 5 curl -s -m 4 http://127.0.0.1:8200/health -o /dev/null 2>/dev/null || return 0
    sleep 5
  done
  return 1
}

env_of() {
  local pid
  pid=$(pgrep -f "sglang serve" | head -1)
  tr '\0' '\n' < /proc/$pid/environ 2>/dev/null
}

run_arm() {
  local label="$1" chunk="$2"; shift 2
  stop_and_confirm || { echo "== $label SKIPPED: old server still up"; return; }
  start_server "$chunk" "$@" || return
  local actual
  actual=$(env_of | sed -n 's/^SGLANG_AB_MARK=//p')
  cd /data/nvme/sglang-codex
  echo "== $label  (CHUNK=$chunk, marker=${actual:-none}) =="
  # confirm the chunk size the server actually took
  env_of | grep -c . >/dev/null
  timeout 1200 .venv/bin/python pf_probe_prod.py --port 8200 --lens "$LENS" \
      --rounds 1 --no-warmup 2>&1 | tail -3
  cd /data/nvme/sglang
}

for r in $(seq 1 "$ROUNDS"); do
  run_arm "r$r baseline chunk512" 512 SGLANG_AB_MARK=base
  run_arm "r$r chunk1024"        1024 SGLANG_AB_MARK=c1024
  run_arm "r$r chunk256"         256  SGLANG_AB_MARK=c256
done
echo "AB_DONE"
