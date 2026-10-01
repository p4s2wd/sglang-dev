#!/bin/bash
# usage: prof_long.sh
# Profile decode at a topk-saturating context so the family ranking reflects the real
# operating point. Every previous decode profile was driven with a one-line prompt,
# which measures short-context decode -- 2.1x faster than the 216K target and with a
# completely different attention shape (a few tiles instead of 32).
#
# 86K prompt tokens is the shortest context where index_topk=512 saturates, so the
# attention shape matches 256K while the prefill costs ~115 s instead of ~845 s.
# Step count is known by construction: one request, 33 tokens, bs=1.
cd /data/nvme/sglang-codex
. ./env.sh
OUT=profiles/long-$(date +%s)
mkdir -p $OUT
timeout 2400 .venv/bin/python - <<PY
import json, random, time, urllib.request
def post(p, pl=None, t=2500):
    r = urllib.request.Request("http://127.0.0.1:30000"+p, data=json.dumps(pl or {}).encode(),
                               headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(r, timeout=t).read().decode()
W=["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel","india","juliet",
   "kilo","lima","mike","november","oscar","papa"]
rnd=random.Random(424242)
txt=" ".join(rnd.choice(W) for _ in range(76000))+" end"
try: post("/stop_profile")
except Exception: pass
# warm the prompt twice so the profiled call is a cached prefill plus decode
post("/generate", {"text":txt,"sampling_params":{"temperature":0.0,"max_new_tokens":1}})
post("/generate", {"text":txt,"sampling_params":{"temperature":0.0,"max_new_tokens":1}})
print(post("/start_profile", {"output_dir":"$OUT","activities":["GPU"],
      "profile_by_stage":True,"num_steps":300,"profile_prefix":"lg"}), flush=True)
j=json.loads(post("/generate", {"text":txt,"sampling_params":{"temperature":0.0,"max_new_tokens":33}}))
print("completion:", j["meta_info"]["completion_tokens"], flush=True)
time.sleep(10)
try: print(post("/stop_profile"))
except Exception as e: print("stop:", type(e).__name__)
PY
sleep 8
echo "OUTDIR=$OUT"
