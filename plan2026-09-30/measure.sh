#!/usr/bin/env bash
# Run the identical measurement suite against whatever server is up.
#
# Both arms MUST go through this exact script: the only thing allowed to
# differ between arms is the flag under test. Anything else -- different round
# counts, a missing warmup, a different prompt set -- silently invalidates an
# interleaved A/B on a box with ~6-10% thermal drift.
#
# Per arm it produces:
#   res/greedy_<TAG>1.json, _<TAG>2.json   two captures, compared (within-instance determinism)
#   res/dec_<TAG>.json                     decode tok/s at bs 1,2,4,8 + graph coverage
#   res/pf_<TAG>.json                      prefill tok/s at lens x concurrency + admission histograms
#
# Usage: ./measure.sh <TAG>
set -uo pipefail
cd /data/nvme/sglang-codex/plan2026-09-30

TAG="${1:?usage: measure.sh <TAG>}"
# Concurrency sweep for the decode probe. Appending larger values does not move
# any existing bs within its round, so N<=8 stays comparable with an arm that
# used the shorter default.
DEC_BS="${DEC_BS:-1,2,4,8}"
PY=/data/nvme/sglang/.venv/bin/python
mkdir -p res

echo "############################################"
echo "# measuring arm TAG=$TAG"
echo "############################################"

if ! curl -s -m 5 -o /dev/null http://127.0.0.1:8200/health; then
  echo "SERVER NOT HEALTHY -- aborting" >&2
  exit 1
fi

# The server must actually carry the flag we think it does, or we would be
# measuring the previous arm under a new label.
SPID="$(pgrep -f 'bin/sglang' | head -1)"
echo "-- server pid: ${SPID:-<none>}"
if [ -n "${SPID:-}" ]; then
  tr '\0' '\n' < "/proc/$SPID/environ" 2>/dev/null \
    | grep -E '^SGLANG_PP_EARLY_PROXY_SEND=' || echo "  (flag unset)"
fi

echo
echo "== [1/4] greedy capture x2 (within-instance determinism) =="
$PY greedy.py capture "res/greedy_${TAG}1.json" --port 8200 --n 6 >/dev/null 2>&1 \
  && echo "  captured ${TAG}1" || echo "  CAPTURE ${TAG}1 FAILED" >&2
$PY greedy.py capture "res/greedy_${TAG}2.json" --port 8200 --n 6 >/dev/null 2>&1 \
  && echo "  captured ${TAG}2" || echo "  CAPTURE ${TAG}2 FAILED" >&2
$PY greedy.py compare "res/greedy_${TAG}1.json" "res/greedy_${TAG}2.json" \
  && echo "  within-instance determinism: OK" \
  || echo "  WITHIN-INSTANCE NONDETERMINISM -- treat parity as unusable" >&2

# Abort on a dead server instead of burning the remaining ~12 minutes. The
# first requests after a cold start are exactly when the prefill Triton kernels
# device-load and when the IMA fault fires (3 of 5 instances in this session),
# so a dead server means this arm produces no data at all. greedy.py compare
# does not exit non-zero on connection errors (both sides become ""), hence the
# explicit health check rather than relying on its exit status.
if ! curl -s -m 5 -o /dev/null http://127.0.0.1:8200/health; then
  echo "  SERVER DIED DURING GREEDY CAPTURE -- aborting this arm" >&2
  echo "  (archive /data/nvme/sglang/logs/serve-prod.log BEFORE restarting)" >&2
  exit 3
fi

echo
echo "== [2/4] decode probe ($DEC_BS x 3 rounds, interleaved) =="
$PY dec_probe.py --bs "$DEC_BS" --newtok 256 --rounds 3 --warmup 2 \
    --json-out "res/dec_${TAG}.json" 2>&1 | tail -14

echo
echo "== [3/4] prefill concurrency probe (lens 1000,4000 x K 1,2,4,8) =="
$PY pf_conc_probe.py --lens 1000,4000 --ks 1,2,4,8 --rounds 3 --warmup 1 \
    --json-out "res/pf_${TAG}.json" 2>&1 | grep -A6 "aggregate prefill"

echo
echo "== [4/4] graph coverage during this arm =="
grep -o "cuda graph: [A-Za-z]*" /data/nvme/sglang/logs/serve-prod.log | sort | uniq -c
grep -o "#running-req: [0-9]*" /data/nvme/sglang/logs/serve-prod.log | sort | uniq -c

echo
echo "MEASURE_DONE tag=$TAG"
