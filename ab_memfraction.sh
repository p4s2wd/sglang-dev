#!/usr/bin/env bash
# Can the fast split (11,11,11,10) be made safe?
#
# The measured problem: with 11,11,11,10, PP0 carries 11 layers instead of 10.
# Per-card weight after TP2 split is 1706 MiB per layer, so PP0's weights go from
# ~17570 to ~19276 MiB on a 22528 MiB card (85.6%). After decode-graph capture PP0
# has only 0.42 GB free versus 0.74 GB for the shipped split, and a 200-question
# GSM8K run died in ncclAllGather with an unhandled CUDA error.
#
# The lever is headroom, not layers. mem-fraction-static 0.97 reserves 97% of the
# card for weights + KV pool, leaving ~3% (about 0.68 GiB) for activations. Lowering
# it shrinks the KV pool and gives activations more room. The cost is KV capacity,
# which is a capacity trade-off rather than a correctness one.
#
# Arms sweep mem-fraction with the fast split held fixed, and read the resulting
# per-rank headroom off the server's own startup log, because that number (not the
# tok/s) is what decided the crash.
#
# Usage: ab_memfraction.sh [rounds]
set -uo pipefail
ROUNDS="${1:-1}"
POOL=210000
PART=11,11,11,10
LAUNCH=/data/nvme/sglang/deepseek-v4-flash.sh
cd /data/nvme/sglang

stop_and_confirm() {
  timeout 200 "$LAUNCH" --stop >/dev/null 2>&1
  for _ in $(seq 1 24); do
    timeout 5 curl -s -m 4 http://127.0.0.1:8200/health -o /dev/null 2>/dev/null || return 0
    sleep 5
  done
  return 1
}

run_arm() {
  local label="$1" frac="$2" pool="$3"
  stop_and_confirm || { echo "== $label SKIPPED (old server up)"; return; }
  setsid nohup env SGLANG_PP_LAYER_PARTITION="$PART" MEM_FRACTION="$frac" \
      MAX_TOTAL_TOKENS="$pool" "$LAUNCH" \
      >/tmp/opencode/mf_restart.log 2>&1 </dev/null &
  disown
  local ok=1
  for _ in $(seq 1 40); do
    sleep 10
    if timeout 5 curl -s -m 4 http://127.0.0.1:8200/health -o /dev/null 2>/dev/null; then ok=0; break; fi
  done
  if [ $ok -ne 0 ]; then
    echo "== $label FAILED TO START (frac=$frac) =="
    grep -oE "CUDA out of memory|OutOfMemoryError|no available memory|ValueError[^\"]*" \
      logs/serve-prod.log 2>/dev/null | sort -u | tail -2
    return
  fi
  local av frac_seen
  av=$(grep -oE "Capture target decode CUDA graph end.*avail mem=[0-9.]+ GB" \
        logs/serve-prod.log 2>/dev/null | grep -oE "avail mem=[0-9.]+ GB" | sort -t= -k2 -n | head -1)
  frac_seen=$(grep -oE "mem_fraction_static': [0-9.]+" logs/serve-prod.log 2>/dev/null | tail -1)
  echo "== $label  (frac=$frac pool=$pool  $frac_seen  min avail: ${av#avail mem=}) =="
  # short throughput check
  cd /data/nvme/sglang-codex
  timeout 900 .venv/bin/python pf_probe_prod.py --port 8200 --lens 13000 \
      --rounds 1 --no-warmup 2>&1 | tail -2
  cd /data/nvme/sglang
}

for r in $(seq 1 "$ROUNDS"); do
  run_arm "r$r frac0.97 pool210k" 0.97  210000
  run_arm "r$r frac0.94 pool180k" 0.94  180000
  run_arm "r$r frac0.90 pool140k" 0.90  140000
done
echo "AB_DONE"
