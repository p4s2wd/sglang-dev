#!/bin/bash
# Profile the server while it decodes, to find where wall time goes when GPU
# utilization is ~7%. py-spy's raw format is one sampled stack per line, so the
# leaf frame (where the CPU actually was) is just the last field.
set -u
cd /data/nvme/sglang-codex
. ./env.sh

# The scheduler processes rename themselves via setproctitle
# ("sglang::scheduler_PP0_TP0"), so py-spy cannot find them by walking the
# launcher's children -- name the pid directly.
MAIN=$(pgrep -f "sglang::scheduler_PP0_TP0" | head -1)
echo "profiling scheduler pid=$MAIN, 20s"

( bash ./probe_generate.sh \
    "Explain in detail how a steam turbine converts heat into rotational work." 200 \
    >/dev/null 2>&1 ) &
GEN=$!
sleep 2

py-spy record --pid "$MAIN" --duration 20 --format raw \
  --output /tmp/sched.raw.txt 2>&1 | tail -2
wait $GEN 2>/dev/null

echo "=== leaf frames (where the CPU actually was) ==="
# raw lines look like:  frame;frame;frame  count
cut -d ' ' -f 1 /tmp/sched.raw.txt 2>/dev/null \
  | awk -F ';' 'NF {print $NF}' \
  | sort | uniq -c | sort -rn | head -15

echo "=== hottest files ==="
cut -d ' ' -f 1 /tmp/sched.raw.txt 2>/dev/null \
  | awk -F ';' 'NF {print $NF}' \
  | sed -E 's/.*\(([^()]+)\)$/\1/' \
  | sort | uniq -c | sort -rn | head -12
