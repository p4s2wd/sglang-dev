"""Attribute the exact capture's decode kernels to sglang call sites.

The exact capture ran with activities=["GPU"] so it has no CPU ops or Python frames,
and decode runs under CUDA graphs so 99% of its kernels have no External id anyway.
The dst capture ran with CPU+GPU and with_stack, and its EXTEND trace runs eager, so
it does carry frames. A kernel's (name, grid, block) triple identifies the op and the
shape it ran on, so build that map from dst's EXTEND trace and look up the exact
capture's decode kernels with it.
"""
import bisect, gzip, glob, json, sys
from collections import defaultdict

EAGER = "profiles/dst-1790054370/*EXTEND*.trace.json.gz"
DEC = "profiles/exact-1790060350/*TP-0-PP-*.trace.json.gz"

fe = max(glob.glob(EAGER))
eve = json.load(gzip.open(fe))["traceEvents"]
py = sorted([e for e in eve if e.get("cat") == "python_function" and ".py(" in e.get("name", "")],
            key=lambda e: e["ts"])
starts = [e["ts"] for e in py]
cpu = sorted([e for e in eve if e.get("cat") == "cpu_op"], key=lambda e: e["ts"])
ext2cpu = {}
for c in cpu:
    eid = c.get("args", {}).get("External id")
    if eid is not None:
        ext2cpu[eid] = c


def frame_of(ts):
    i = bisect.bisect_right(starts, ts)
    best = None
    for e in py[max(0, i - 1500):i]:
        if e["ts"] <= ts <= e["ts"] + e.get("dur", 0):
            n = e["name"]
            if "/site-packages/" in n:
                continue
            if "/sglang/" in n:
                best = n
    return best


t2f = {}
for e in eve:
    if e.get("cat") != "kernel":
        continue
    eid = e.get("args", {}).get("External id")
    if eid is None:
        continue
    c = ext2cpu.get(eid)
    if c is None:
        continue
    fr = frame_of(c["ts"])
    if not fr:
        continue
    t2f.setdefault((e["name"][:70], tuple(e["args"].get("grid", [])),
                    tuple(e["args"].get("block", []))), (fr, c["name"]))
print("eager %s: %d (kernel,grid,block) -> frame entries" % (fe.split("/")[-1][:36], len(t2f)))

agg = defaultdict(lambda: [0.0, 0])
for f in sorted(glob.glob(DEC)):
    if "EXTEND" in f:
        continue
    ev = json.load(gzip.open(f))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    steps = max(1, sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]) / 22.0)
    for e in ks:
        n = e["name"]
        if not ("elementwise" in n or "reduce_kernel" in n or "fill" in n):
            continue
        key = (n[:70], tuple(e["args"].get("grid", [])), tuple(e["args"].get("block", [])))
        hit = t2f.get(key)
        lab = (hit[0] + " [aten:" + hit[1] + "]") if hit else ("UNMATCHED " + n[:50])
        agg[lab][0] += e["dur"] / 1e3 / steps
        agg[lab][1] += 1

tot = sum(v[0] for v in agg.values())
print("\n%.2f ms/token of elementwise/reduce/fill across the 4 stages" % tot)
print("%9s %9s  call site" % ("ms/tok", "calls/tok"))
for k, (ms, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:14]:
    print("%9.3f %9d  %s" % (ms, n, k[:118]))
