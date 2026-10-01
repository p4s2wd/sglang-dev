"""Name the elementwise kernels precisely enough to fuse them.

Attention (31.2%) is at Triton's operand-staging floor -- only hand-written mma
goes further, which is a multi-day rewrite. The elementwise group is the other
11% and is 4800 separate launches, which is the objective's third item (fuse the
per-layer small kernels). But the earlier ranking truncated names to 58 chars, so
the same at::native::vectorized_elementwise_kernel line covers dozens of distinct
lambdas. This groups by the full demangled name and reports calls and per-call
time, which is what identifies a fusion target: many calls of the same tiny op
per layer is a chain that should be one kernel.
"""
import gzip, glob, json, sys, re
from collections import defaultdict

d = sys.argv[1]
f = max(glob.glob(d + "/*PP-3*EXTEND*"))
ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]
ks = [e for e in ev if e.get("cat") == "kernel"]
agg = defaultdict(lambda: [0.0, 0])
for e in ks:
    n = e["name"]
    # Keep the functor identity: the <...> template args carry the op.
    agg[n[:150]][0] += e["dur"] / 1e3
    agg[n[:150]][1] += 1
tot = sum(v[0] for v in agg.values())
print("total %.1f ms; elementwise/copy kernels ranked by total time" % tot)
print("%8s %6s %8s  name" % ("ms", "calls", "us/call"))
pat = re.compile(r"elementwise|vectorized|unrolled|Copy|copy|CatArray|index_|fill_|reduce_kernel|triton_poi|triton_red")
sel = [(v[0], v[1], k) for k, v in agg.items() if pat.search(k)]
for ms, ncalls, name in sorted(sel, reverse=True)[:14]:
    print("%8.1f %6d %8.1f  %s" % (ms, ncalls, ms * 1e3 / ncalls, name[:118]))
print("\nsum of elementwise-class kernels: %.1f ms (%.1f%%)"
      % (sum(s[0] for s in sel), 100 * sum(s[0] for s in sel) / tot))
