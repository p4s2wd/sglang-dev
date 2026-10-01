#!/usr/bin/env bash
# Stop, start, and wait for the production server, then echo the memory lines
# that decide whether an arm is viable.
#
# Every A/B arm needs its own restart: CUDA graph capture memory, the KV pool
# sizing and the Triton JIT cache are all decided at boot, and this box's rule
# is "restart before you push load". So this is the unit of the experiment.
#
# deepseek-v4-flash.sh truncates logs/serve-prod.log on start, so everything
# we read afterwards is this arm's log by construction -- no offsets needed.
#
# Usage:
#   ./ab_restart.sh                                   # current defaults
#   GRAPH_BS="1 2 4 8" GRAPH_MAX_BS=8 ./ab_restart.sh
set -uo pipefail

LAUNCH=/data/nvme/sglang/deepseek-v4-flash.sh
LOG=/data/nvme/sglang/logs/serve-prod.log
PORT=8200

echo "== stopping =="
timeout 300 "$LAUNCH" --stop || true

if pgrep -f "sglang serve" >/dev/null; then
  echo "still alive after --stop:" >&2
  pgrep -af "sglang serve" >&2
  exit 1
fi

echo "== starting =="
echo "   GRAPH_BS='${GRAPH_BS:-<default>}'  GRAPH_MAX_BS='${GRAPH_MAX_BS:-<default>}'  PREFILL_GRAPH='${PREFILL_GRAPH:-disabled}'"

cd /data/nvme/sglang
# 120 s was enough for boot + decode capture (~74 s). A prefill graph adds a
# capture per token bucket (29 buckets at max_bs=512), so the launcher needs
# far more room -- otherwise `timeout` SIGTERMs the daemon mid-capture and the
# arm reports "SERVER DIED" for a reason that has nothing to do with the flag.
timeout 900 env \
  ${GRAPH_BS:+GRAPH_BS="$GRAPH_BS"} \
  ${GRAPH_MAX_BS:+GRAPH_MAX_BS="$GRAPH_MAX_BS"} \
  ${MAX_TOTAL_TOKENS:+MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS"} \
  ${MAXREQ:+MAXREQ="$MAXREQ"} \
  ${MEM_FRACTION:+MEM_FRACTION="$MEM_FRACTION"} \
  ${PREFILL_GRAPH:+PREFILL_GRAPH="$PREFILL_GRAPH"} \
  ${GRAPH_BACKEND_DECODE:+GRAPH_BACKEND_DECODE="$GRAPH_BACKEND_DECODE"} \
  ${PREFILL_GRAPH_MAX_BS:+PREFILL_GRAPH_MAX_BS="$PREFILL_GRAPH_MAX_BS"} \
  ${PREFILL_GRAPH_BS:+PREFILL_GRAPH_BS="$PREFILL_GRAPH_BS"} \
  ${PREFILL_GRAPH_MAX_CONTEXT:+PREFILL_GRAPH_MAX_CONTEXT="$PREFILL_GRAPH_MAX_CONTEXT"} \
  ${SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB:+SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB="$SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB"} \
  ${SGLANG_SM75_HEADSHARED_MIN_BATCH:+SGLANG_SM75_HEADSHARED_MIN_BATCH="$SGLANG_SM75_HEADSHARED_MIN_BATCH"} \
  ${CUDA_LAUNCH_BLOCKING:+CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING"} \
  ${SGLANG_CUDA_COREDUMP:+SGLANG_CUDA_COREDUMP="$SGLANG_CUDA_COREDUMP"} \
  ${SGLANG_CUDA_COREDUMP_DIR:+SGLANG_CUDA_COREDUMP_DIR="$SGLANG_CUDA_COREDUMP_DIR"} \
  ./deepseek-v4-flash.sh || exit 1

for i in $(seq 1 144); do
  sleep 5
  if curl -s -m 4 "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null; then
    echo "== healthy after $(( (i - 1) * 5 ))s =="
    break
  fi
  if ! pgrep -f "sglang serve" >/dev/null; then
    echo "SERVER DIED" >&2
    tail -60 "$LOG" >&2
    exit 1
  fi
done

if ! curl -s -m 4 "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null; then
  echo "NEVER BECAME HEALTHY" >&2
  tail -60 "$LOG" >&2
  exit 1
fi

echo
echo "== memory accounting for this arm =="
grep -E "DSV4 memory calculation|DSV4 pool sizes|Memory pool end|Capture target decode CUDA graph (begin|end)|max_total_num_tokens=|avail mem=" "$LOG" \
  | sed 's/^\[[^]]* //' | sort -u

echo
echo "== flags the server actually carries =="
# Deliberately NOT `pgrep -f "sglang serve"`: the launcher's stop() does
# `pkill -9 -f "sglang serve"`, which matches any process whose *cmdline*
# contains that substring -- including the shell that invoked this script if
# the caller echoed the same pattern. That killed the caller twice before the
# mistake was found. Match on the binary path instead.
SPID="$(pgrep -f 'bin/sglang' | head -1)"
if [ -n "${SPID:-}" ]; then
  tr '\0' '\n' < "/proc/$SPID/environ" 2>/dev/null \
    | grep -E '^SGLANG_PP_EARLY_PROXY_SEND=|^SGLANG_PP_LAYER_PARTITION=|^SGLANG_OPT_W8A16|^SGLANG_DSV4|^SGLANG_TRITON_LOAD|^SGLANG_SM75_|^SGLANG_CUDA_COREDUMP=|^SGLANG_TRITON_SYNC_EVERY_LAUNCH=|^SGLANG_VALIDATE_SPARSE_INDICES=|^PREFILL_GRAPH|^GRAPH_BS=|^GRAPH_MAX_BS=|^GRAPH_BACKEND_DECODE=|^CUDA_LAUNCH_BLOCKING=' \
    | sort
else
  echo "  (no server pid found)"
fi

echo
echo "== OOM / failure check =="
if grep -qi "out of memory\|OutOfMemoryError" "$LOG"; then
  echo "OOM DETECTED IN THIS ARM" >&2
  grep -i -m3 "out of memory" "$LOG" >&2
  exit 2
fi
echo "ok"
