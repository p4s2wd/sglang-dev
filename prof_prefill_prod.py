#!/usr/bin/env python
"""Capture a server-side EXTEND (prefill) profile from the PRODUCTION server.

Why server-side: torch.profiler in the client captures zero kernels, because
the kernels live in the sglang scheduler worker processes. The profile has to be
requested from the server via /start_profile.

Why this is not prof_pf2.sh: that one targets :30000 (the dev server, now
dead) and was captured on 2026-09-20, BEFORE opt1's fused kernels shipped.
The numbers everyone quotes from those traces describe a build that is no
longer what production runs.

The prompts carry a fresh uuid4 at position 0 so the radix prefix cache cannot
serve them -- otherwise we profile cache lookups, not prefill.

Usage:
  prof_prefill_prod.py                      # 3 prompts ~12K tok, port 8200
  prof_prefill_prod.py --port 30000 --words 13000
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
import uuid


def post(port, path, body=None, timeout=900, host="127.0.0.1"):
    data = json.dumps(body).encode() if body is not None else b"{}"
    req = urllib.request.Request(
        f"http://{host}:{port}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default="")
    ap.add_argument("--words", type=int, default=9000,
                    help="filler words per prompt (~1.3 tok/word)")
    ap.add_argument("--n", type=int, default=3, help="prompts to profile")
    ap.add_argument("--num-steps", type=int, default=2,
                    help="prefill batches to capture per stage (see NOTE)")
    ap.add_argument("--no-stack", action="store_true", default=True)
    ap.add_argument("--by-stage", action="store_true", default=True)
    a = ap.parse_args()

    # NOTE on --num-steps: it is NOT optional when profile_by_stage is set.
    # sglang's profiler_manager only initialises the per-stage counters inside
    # `if num_steps:` (profiler_manager.py:144-147); they are declared None in
    # __init__ (line 74-75). So profile_by_stage=True with num_steps=0 leaves
    # profiler_prefill_ct as None and _profile_batch_predicate dies on
    # `self.profiler_prefill_ct += 1` (line 417), taking the whole server down.
    # This is an upstream bug, reproduced here on purpose so it is on record.
    if a.by_stage and not a.num_steps:
        raise SystemExit(
            "refusing to start: profile_by_stage=True with num_steps=0 crashes\n"
            "sglang (TypeError: NoneType + int at profiler_manager.py:417).\n"
            "Pass --num-steps N (N>0)."
        )

    root = "/data/nvme/sglang-codex"
    out = a.out or f"{root}/profiles/pfprod-{int(time.time())}"
    os.makedirs(out, exist_ok=True)

    try:
        post(a.port, "/stop_profile", timeout=120)
    except Exception:
        pass

    # warmup on an unrelated short prompt: JIT, allocator, cuda graph capture
    post(a.port, "/generate", {"text": "warmup probe",
                               "sampling_params": {"temperature": 0.0,
                                                   "max_new_tokens": 1}})

    body = {
        "output_dir": out,
        "activities": ["GPU"],
        "record_shapes": False,
        "with_stack": a.no_stack,
        "profile_prefix": "pf",
    }
    if a.by_stage:
        body["profile_by_stage"] = True
    if a.num_steps:
        body["num_steps"] = a.num_steps
    print(post(a.port, "/start_profile", body), flush=True)

    for i in range(a.n):
        salt = uuid.uuid4().hex
        txt = f"{salt} " + " ".join(["word"] * a.words)
        t0 = time.time()
        o = json.loads(post(a.port, "/generate", {
            "text": txt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        }, timeout=3600))
        dt = time.time() - t0
        n = o["meta_info"]["prompt_tokens"]
        print(f"prompt {i}: {n} tok in {dt:.1f}s = {n/dt:.1f} tok/s", flush=True)

    # the profiler flushes on stop; give the 8 workers time to write
    time.sleep(10)
    try:
        print(post(a.port, "/stop_profile", timeout=300))
    except Exception as e:
        print("stop:", type(e).__name__, e)
    time.sleep(8)

    print(f"\nOUTDIR={out}")
    subprocess.run(["ls", "-la", out])


if __name__ == "__main__":
    main()
