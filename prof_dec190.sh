#!/bin/bash
# usage: prof_dec190.sh
# Re-derive the decode kernel ranking at the current 190 W setting. The ranking
# used so far came from profiles/20260920-123114-decfloor, taken in an earlier
# session at 150 W, and its absolute numbers no longer reconcile: it reports
# 299.9 ms of device time with 22 allreduce calls, which cannot describe a 64
# ms/token decode step. Ratios may still hold, but decisions should not rest on
# them. Capture a fresh DECODE trace on the running production server, with
# enough steps to average over, then rank.
set -u
cd /data/nvme/sglang-codex
. ./env.sh
OUT=profiles/dec190-$(date +%s)
mkdir -p $OUT
timeout 900 .venv/bin/python - <<PY
import json, time, urllib.request
def post(path, payload=None):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request("http://127.0.0.1:30000"+path, data=data,
                                 headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(req, timeout=900).read().decode()
try: post("/stop_profile")
except Exception: pass
# Warm decode so the graph is captured and clocks are up before measuring.
post("/generate", {"text":"warm up please ", "sampling_params":
                   {"temperature":0.0,"max_new_tokens":8}})
print(post("/start_profile", {"output_dir":"$OUT","activities":["GPU"],
      "profile_by_stage":True,"record_shapes":False,"with_stack":False,
      "num_steps":4,"profile_prefix":"d190"}), flush=True)
# Two concurrent requests: bs=2 is the largest batch the user's 1-2 request
# constraint allows, and it is the configuration closest to the 50 tok/s target.
import concurrent.futures as cf
def one(i):
    return json.loads(post("/generate", {"text":"Explain why the sky is blue in detail %d" % i,
          "sampling_params":{"temperature":0.0,"max_new_tokens":48}}))["meta_info"]
with cf.ThreadPoolExecutor(max_workers=2) as ex:
    ms = list(ex.map(one, range(2)))
print("completion tokens:", [m["completion_tokens"] for m in ms], flush=True)
time.sleep(12)
try: print(post("/stop_profile"))
except Exception as e: print("stop:", type(e).__name__)
PY
sleep 8
echo "OUTDIR=$OUT"
