#!/usr/bin/env bash
# Final, like-for-like A/B of the two layer splits at ONE pool size, alternating
# so thermal drift cannot favour either arm.
#
# Why this exists: the +8.8% for 11,11,11,10 was measured across two separate
# runs (1417.4 -> 1543.4 and 1418.7 -> 1543.4) that both used pool=270000, where
# the fast split OOMs PP0 at 0.42 GB free. Running a 200-question eval on it
# died in ncclAllGather. Shrinking the pool to 210000 restores PP0 to 0.68 GB --
# the same headroom the shipped split has -- and a full eval then completed at
# 94.0% with 0 invalid. But a throughput reading taken straight after that eval
# came out at 1260 tok/s because the box was hot, and after a partial cooldown
# 1433.9 -- neither comparable to the 1545.6 measured on a cold box.
#
# So: same pool (210000) for both arms, alternate A/B/A/B, read the env back off
# the running process, and take medians. That is the only comparison that
# isolates the split from the pool size and the thermal state.
#
# Usage: ab_pp_final.sh [rounds]
set -uo pipefail
ROUNDS="${1:-2}"
LENS="8000,13000"
POOL=210000
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
    setsid nohup env MAX_TOTAL_TOKENS="$POOL" "$LAUNCH" \
        >/tmp/opencode/pf_restart.log 2>&1 </dev/null &
  else
    setsid nohup env SGLANG_PP_LAYER_PARTITION="$part" MAX_TOTAL_TOKENS="$POOL" "$LAUNCH" \
        >/tmp/opencode/pf_restart.log 2>&1 </dev/null &
  fi
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
  local av got
  av=$(grep -oE "Capture target decode CUDA graph end.*avail mem=[0-9.]+ GB" \
        logs/serve-prod.log 2>/dev/null | grep -oE "avail mem=[0-9.]+ GB" | sort -t= -k2 -n | head -1)
  got=$(env_of | sed -n 's/^SGLANG_PP_LAYER_PARTITION=//p')
  echo "== $label  (split=${got:-default} pool=$POOL  min avail: ${av#avail mem=}) =="
  cd /data/nvme/sglang-codex
  timeout 1200 .venv/bin/python pf_probe_prod.py --port 8200 --lens "$LENS" \
      --rounds 1 --no-warmup 2>&1 | tail -3
  cd /data/nvme/sglang
}

for r in $(seq 1 "$ROUNDS"); do
  run_arm "r$r A ship 10,11,11,11" default
  run_arm "r$r B fast 11,11,11,10" 11,11,11,10
done
echo "AB_DONE"
