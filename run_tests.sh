#!/bin/bash
# Run the SM75 kernel test suite on the target machine.
# Usage: ./run_tests.sh [gpu_index]
set -u
cd /data/nvme/sglang-codex
. ./env.sh
export CUDA_VISIBLE_DEVICES="${1:-0}"
A=/data/nvme/sglang-codex/sglang-audit

TESTS="
test_mxfp4_w4a16.py
test_moe_sub80_e2e.py
test_mqa_logits.py
test_mqa_logits_real_shape.py
test_jit_store_real.py
test_jit_qindexer_real.py
b8/test_w4a16_ptx.py
b8/test_direct.py
b8/test_e2e_ptx.py
test_real_weights.py
"

fails=0
for t in $TESTS; do
  echo "##### $t"
  if timeout 2400 python "$A/$t" > "/tmp/t_out.log" 2>&1; then
    tail -4 /tmp/t_out.log
    echo "----- PASS $t"
  else
    rc=$?
    tail -20 /tmp/t_out.log
    echo "----- FAIL(rc=$rc) $t"
    fails=$((fails + 1))
  fi
done
echo "SUITE_DONE fails=$fails"
