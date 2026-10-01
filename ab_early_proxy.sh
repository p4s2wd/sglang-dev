#!/usr/bin/env bash
# A/B SGLANG_PP_EARLY_PROXY_SEND on the production topology.
#
# Why this is the one comm-side lever worth testing:
#
# Measured facts on this box (profile of the running prod server):
#   * P2P is only available WITHIN (0,1)(2,3)(4,5)(6,7); every cross-pair hop is NS.
#   * TP2's all-reduce therefore lands inside an NVLink pair on all 4 stages --
#     already optimal, nothing to gain there.
#   * The PP handoff crosses pairs, but moves one 2 MB proxy tensor per chunk, and
#     its nccl SendRecv overlaps ~100% with other kernels: it is a SPIN on the
#     previous stage, not a transfer. So the link is not the cost.
#   * The cost is the ORDERING: in scheduler_pp_mixin.py the proxy send is issued
#     AFTER the blocking output recv + d2h_event.synchronize() + the CPU work in
#     process_batch_result. So every stage's CPU tail lands on the downstream
#     critical path. SGLANG_PP_EARLY_PROXY_SEND reorders it to issue the send
#     right after the forward, overlapping that tail with compute.
#
# The env var is read at scheduler init, so each arm needs a server restart.
# Arms alternate to cancel thermal drift.
#
# Usage: ab_early_proxy.sh [rounds] [lens]
set -uo pipefail

ROUNDS="${1:-2}"
LENS="${2:-8000,13000}"
cd /data/nvme/sglang

start_server() {
  local early="$1"
  # Absolute path: the previous version used "./deepseek-v4-flash.sh" with no cd,
  # so `env` failed with "No such file or directory" and NO server was started.
  # The health check then passed against the PREVIOUS arm's still-running server,
  # which silently produced garbage numbers for arm B. Use the absolute path and
  # fail loudly if the launcher does not come up.
  setsid nohup env SGLANG_PP_EARLY_PROXY_SEND="$early" \
    /data/nvme/sglang/deepseek-v4-flash.sh \
    >/tmp/opencode/ep_restart.log 2>&1 </dev/null &
  disown
  sleep 3
  if ! grep -q "No such file" /tmp/opencode/ep_restart.log 2>/dev/null; then :; else
    echo "launcher failed to exec: $(tail -1 /tmp/opencode/ep_restart.log)" >&2
    return 1
  fi
  for _ in $(seq 1 40); do
    sleep 10
    if timeout 5 curl -s -m 4 http://127.0.0.1:8200/health -o /dev/null 2>/dev/null; then
      return 0
    fi
  done
  echo "SERVER FAILED TO START (early=$early)" >&2
  return 1
}

stop_server() { timeout 200 /data/nvme/sglang/deepseek-v4-flash.sh --stop >/dev/null 2>&1; }

echo "arms: A = EARLY_PROXY_SEND=0 (current)  B = EARLY_PROXY_SEND=1"
for r in $(seq 1 "$ROUNDS"); do
  for arm in A B; do
    if [ "$arm" = A ]; then early=0; else early=1; fi
    stop_server
    # Refuse to measure if anything is still up: otherwise a failed start would
    # silently measure the previous arm's server.
    for _ in $(seq 1 20); do
      timeout 5 curl -s -m 4 http://127.0.0.1:8200/health -o /dev/null 2>/dev/null || break
      sleep 5
    done
    if timeout 5 curl -s -m 4 http://127.0.0.1:8200/health -o /dev/null 2>/dev/null; then
      echo "== round $r arm $arm SKIPPED: old server still listening ==" >&2
      continue
    fi
    start_server "$early" || exit 1
    # Confirm the server process itself carries the flag, and print what it
    # actually is. The previous version of this check only counted matches,
    # which printed 1 for "set to 0" and 0 for "not set" -- indistinguishable
    # from "set to 1" / "set to 0" at a glance and easy to misread as inverted.
    cd /data/nvme/sglang-codex
    actual=$(pid=$(pgrep -f "sglang serve" | head -1); \
             tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | \
             sed -n 's/^SGLANG_PP_EARLY_PROXY_SEND=//p')
    echo "== round $r arm $arm (want=$early, server has=${actual:-<unset>}) =="
    if [ "$actual" != "$early" ]; then
      echo "   WARNING: flag mismatch, skipping this arm" >&2
      continue
    fi
    timeout 1200 .venv/bin/python pf_probe_prod.py --port 8200 --lens "$LENS" \
        --rounds 1 --no-warmup 2>&1 | tail -3
  done
done
echo "AB_DONE"
