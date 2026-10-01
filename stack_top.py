"""Name the elementwise kernels by their Python call site.

Elementwise/copy work is 13.6% of prefill device time over ~4800 launches, but the
kernel-name ranking cannot identify a fusion target: one
at::native::vectorized_elementwise_kernel line covers dozens of distinct lambdas.
Prefill runs eager, so a with_stack capture carries each kernel's Python frames.
This attributes device time to the deepest sglang frame in each stack, which is
the line to change. Ops running in fp32 on an fp16 stack are a special flag --
they move twice the bytes for no benefit.
"""
import gzip, glob, json, sys, re
from collections import defaultdict

d = sys.argv[1]
f = max(glob.glob(d + "/*PP-3*EXTEND*"))
ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]
by_id = {e.get("args", {}).get("correlation_id", id(e)): e for e in ev}

# Map kernels to their python stack via the profiler's external-id linkage.
ext = defaultdict(list)
for e in ev:
    if e.get("cat") == "python_function" or "Name" in e.get("name", ""):
        pass

pat = re.compile(r"elementwise|vectorized|unrolled|Copy|copy|index_|fill_|reduce_kernel|pow_|silu")
agg = defaultdict(lambda: [0.0, 0])
stacks = defaultdict(lambda: [0.0, 0])
# torch profiler nests python_function events; use ts containment instead of ids.
py = sorted([e for e in ev if e.get("cat") == "python_function"], key=lambda e: e["ts"])
starts = [e["ts"] for e in py]

def find_frame(ts):
    """Innermost python frame containing ts, restricted to sglang code."""
    import bisect
    i = bisect.bisect_right(starts, ts) - 1
    best = None
    for e in py[max(0, i - 400):i + 1]:
        if e["ts"] <= ts <= e["ts"] + e.get("dur", 0):
            fn = e.get("args", {}).get("module_path") or e.get("args", {}).get("file_path") or ""
            if "/sglang/" in fn and "/site-packages/" not in fn:
                best = "%s:%s %s" % (fn.split("/sglang/")[-1], e.get("args", {}).get("line_no", "?"),
                                     e.get("name", "")[:34])
    return best

ks = [e for e in ev if e.get("cat") == "kernel" and pat.search(e["name"])]
ks.sort(key=lambda e: e["ts"])
tot = sum(e["dur"] for e in ev if e.get("cat") == "kernel") / 1e3
for e in ks[:1200]:
    fr = find_frame(e["ts"])
    key = fr or e["name"][:60]
    stacks[key][0] += e["dur"] / 1e3
    stacks[key][1] += 1

print("total device %.1f ms; elementwise-class device time by sglang call site" % tot)
print("%8s %6s %8s  call site" % ("ms", "calls", "us/call"))
for k, (ms, n) in sorted(stacks.items(), key=lambda x: -x[1][0])[:16]:
    print("%8.1f %6d %8.1f  %s" % (ms, n, ms * 1e3 / n, k[:104]))
