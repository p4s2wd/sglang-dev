"""Attribute DECODE elementwise kernels to sglang call sites by joining on
(kernel name, grid, block) against an eager PREFILL trace.

Decode runs under CUDA graphs, so 4269 of 4330 kernels in a decode trace carry no
External id and cannot be linked to a CPU op or a Python frame -- stack attribution
is impossible for them. But the per-layer code is identical in prefill, which runs
eager and does have full stacks. A kernel's name plus its launch geometry is a
strong identity: the same op on the same tensor shape produces the same triple.

So: build (name, grid, block) -> innermost sglang Python frame from the eager
prefill trace, then look up decode kernels by the same triple. Unmatched decode
entries are reported separately rather than guessed at, since decode-only shapes
(batch 1-2 rows) legitimately have no prefill counterpart.
"""
import bisect, gzip, glob, json, sys
from collections import defaultdict

eager_dir, dec_dir = sys.argv[1], sys.argv[2]


def load(pat):
    hits = glob.glob(pat) or glob.glob(pat.replace(chr(42)+"DECODE", ""))
    if not hits: raise SystemExit("no trace matching " + pat)
    f = max(hits)
    return f, json.load(gzip.open(f))["traceEvents"]


# --- 1. eager prefill: triple -> sglang frame ---
fe, eve = load(eager_dir + "/*EXTEND*.trace.json.gz")
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
    for e in py[max(0, i - 1200):i]:
        if e["ts"] <= ts <= e["ts"] + e.get("dur", 0):
            n = e["name"]
            if "/site-packages/" in n:
                continue
            if "/sglang/" in n:
                best = n
    return best


triple2frame = {}
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
    key = (e["name"][:70], tuple(e["args"].get("grid", [])), tuple(e["args"].get("block", [])))
    triple2frame.setdefault(key, (fr, c["name"]))
print("eager trace %s: %d distinct (kernel,grid,block) -> frame entries"
      % (fe.split("/")[-1][:40], len(triple2frame)))

# --- 2. decode: group by triple, look up ---
fd, edv = load(dec_dir + "/*DECODE*.trace.json.gz")
agg = defaultdict(lambda: [0.0, 0])
for e in edv:
    if e.get("cat") != "kernel":
        continue
    n = e["name"]
    if not any(p in n for p in ("elementwise", "reduce_kernel", "copy_kernel", "fill_",
                                "pow_", "index_", "Cat", "gemvx")):
        continue
    key = (n[:70], tuple(e["args"].get("grid", [])), tuple(e["args"].get("block", [])))
    hit = triple2frame.get(key)
    label = (hit[0] + "  [aten:" + hit[1] + "]") if hit else ("UNMATCHED " + n[:56])
    agg[label][0] += e["dur"] / 1e3
    agg[label][1] += 1

tot = sum(v[0] for v in agg.values())
print("decode trace %s: %.1f ms of elementwise/gemv-class device time" % (fd.split("/")[-1][:40], tot))
print("\n%8s %6s %8s  call site" % ("ms", "calls", "us/call"))
for k, (ms, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:16]:
    print("%8.1f %6d %8.1f  %s" % (ms, n, ms * 1e3 / n, k[:112]))
