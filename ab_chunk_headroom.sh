#!/usr/bin/env bash
# Does buying prefill headroom (by shrinking the KV pool) let a larger chunk work,
# and is a larger chunk actually faster?
#
# Measured facts this follows from:
#   * The server starts with only 0.49-0.61 GiB free per card. max-total-tokens is
#     270000 at mem-fraction 0.97, so the KV pool eats everything left.
#   * chunk=1024 at max-total-tokens=270000 OOMs (reproduced 2026-09-29).
#   * Per-stage compute is 282 ms but the wall is 372 ms/chunk, i.e. ~90 ms
#     (24%) is per-chunk pipeline bubble. A larger chunk amortises that bubble
#     over twice the tokens, so it should be worth ~12% if it runs at all.
#
# So the test is: shrink the KV pool just enough to buy headroom, then raise the
# chunk. Arms alternate; each arm's env is read back off the running process
# before measuring, because a failed start otherwise measures the previous arm.
#
# Usage: ab_chunk_headroom.sh [rounds]
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

# run_arm <label> <CHUNK> <MAX_TOTAL_TOKENS>
run_arm() {
  local label="$1" chunk="$2" mtt="$3"
  stop_and_confirm || { echo "== $label SKIPPED (old server up)"; return; }
  setsid nohup env CHUNK="$chunk" MAX_TOTAL_TOKENS="$mtt" "$LAUNCH" \
      >/tmp/opencode/ch_restart.log 2>&1 </dev/null &
  disown
  local ok=1
  for _ in $(seq 1 40); do
    sleep 10
    if timeout 5 curl -s -m 4 http://127.0.0.1:8200/health -o /dev/null 2>/dev/null; then ok=0; break; fi
  done
  if [ $ok -ne 0 ]; then
    echo "== $label FAILED TO START =="
    grep -oE "CUDA out of memory|OutOfMemoryError|free device mem: [0-9.]+ GiB" \
      logs/serve-prod.log 2>/dev/null | sort -u | tail -3
    return
  fi
  # read the actual free memory the server reported at startup
  local free
  free=$(grep -oE "free device mem: [0-9.]+ GiB" logs/serve-prod.log 2>/dev/null | tail -1)
  echo "== $label  (CHUNK=$chunk MAX_TOTAL_TOKENS=$mtt  $free) =="
  cd /data/nvme/sglang-codex
  timeout 1200 .venv/bin/python pf_probe_prod.py --port 8200 --lens "$LENS" \
      --rounds 1 --no-warmup 2>&1 | tail -3
  cd /data/nvme/sglang
}

for r in $(seq 1 "$ROUNDS"); do
  run_arm "r$r A baseline 512/270000"  512  270000
  run_arm "r$r B 1024/230000"        1024  230000
  run_arm "r$r C 768/250000"          768  250000
done
echo "AB_DONE"
