#!/bin/bash
# Release verification against the ALREADY RUNNING committed-defaults server.
# Completes the verify_fused.sh items interrupted by the outage (10e8e6ad85 fused merge).
cd /data/nvme/sglang-codex
. ./env.sh
echo "=== short context decode (baseline 17.6-18.6) ==="
for r in 1 2 3; do timeout 400 bash probe_perf.sh 4 64 2>&1 | grep -a "decode throughput"; done
echo "=== decode @86K (baseline 13.55) ==="
timeout 1500 .venv/bin/python dec_ctx_ab.py 2>&1 | grep -aE "ctx|FAIL"
echo "=== decode @216K (baseline 13.26) ==="
timeout 2400 .venv/bin/python dec_ctx3.py 150000 2>&1 | tail -2
echo "=== prefill rates (baseline 1052/1053/960) ==="
timeout 900 .venv/bin/python probe_prefill_rate2.py 2>&1 | tail -6
echo RELEASE_VERIFY_DONE
