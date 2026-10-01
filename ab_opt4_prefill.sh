#!/usr/bin/env bash
# Honest A/B for the opt4 moe_combine fix: alternate the two variants across
# rounds and take the median per arm, so the box's ~6-10% thermal drift cannot
# masquerade as a speedup.
#
# Why this exists: the single-pass before/after numbers drifted with the machine.
# The first "after" run read 1429 tok/s and a later one read 1366 tok/s on the
# same build, because the GPU had been under profiling load for hours. Only an
# interleaved comparison supports a claim.
#
# Arms:
#   A = opt3 elementwise.py (the O(num_slots) scan)
#   B = opt4 elementwise.py (the slot table)
#
# Usage: ab_opt4_prefill.sh [rounds] [lens]
set -uo pipefail

ROUNDS="${1:-2}"
LENS="${2:-8000,13000}"
SP=/data/nvme/sglang/.venv/lib/python3.12/site-packages/sglang/kernels/ops/elementwise/elementwise.py
CH=/data/nvme/sglang-codex/sglang/python/sglang/kernels/ops/elementwise/elementwise.py
OPT3=/tmp/opencode/elementwise.py.opt3
OUT=/tmp/opencode/ab_opt4
mkdir -p "$OUT"

if [ ! -f "$OPT3" ]; then
  echo "missing $OPT3 (the opt3 copy of elementwise.py)" >&2; exit 1
fi

restore() { cp "$CH" "$SP"; }
trap restore EXIT

start_server() {
  cd /data/nvme/sglang
  setsid nohup ./deepseek-v4-flash.sh >/tmp/opencode/ab_restart.log 2>&1 </dev/null &
  disown
  for _ in $(seq 1 40); do
    sleep 10
    if timeout 5 curl -s -m 4 http://127.0.0.1:8200/health -o /dev/null 2>/dev/null; then
      return 0
    fi
  done
  echo "SERVER FAILED TO START" >&2
  return 1
}

stop_server() {
  cd /data/nvme/sglang
  timeout 200 ./deepseek-v4-flash.sh --stop >/dev/null 2>&1
}

for r in $(seq 1 "$ROUNDS"); do
  for arm in A B; do
    if [ "$arm" = A ]; then cp "$OPT3" "$SP"; else cp "$CH" "$SP"; fi
    stop_server
    start_server || exit 1
    cd /data/nvme/sglang-codex
    echo "== round $r arm $arm =="
    timeout 1200 .venv/bin/python pf_probe_prod.py --port 8200 --lens "$LENS" \
        --rounds 1 --no-warmup 2>&1 | tail -3
  done
done

restore
echo "AB_DONE"
