"""Why do identical attention launches take 2 ms or 25 ms in the same trace?

The prefill trace shows 88 launches of the same kernel at the same grid spanning
2.06 to 25.31 ms. The kernel's own work cannot vary 10x at a fixed grid and a
fixed topk, so something external is stealing the SMs. The prime suspect is the
PP pipeline: ncclDevKernel_SendRecv runs proxy transfers on the same GPU
concurrently with compute (17.2 ms x 9 calls in this trace), and NCCL kernels
occupy SMs for their whole duration. If the slow attention calls overlap nccl
and the fast ones do not, the lever is scheduling, not the kernel -- and
optimizing the kernel would recover far less than its 39.6% share suggests.
"""
import gzip, glob, json, sys

d = sys.argv[1]
f = max(glob.glob(d + "/*TP-0-PP-3*EXTEND*"))
ev = json.load(gzip.open(f))["traceEvents"]
ks = [e for e in ev if e.get("cat") == "kernel"]

hs = [e for e in ks if "headshared" in e["name"]]
nccl = [e for e in ks if "nccl" in e["name"].lower() or "all_reduce" in e["name"]]
nccl.sort(key=lambda e: e["ts"])

def overlap_ms(a, b):
    lo = max(a["ts"], b["ts"])
    hi = min(a["ts"] + a["dur"], b["ts"] + b["dur"])
    return max(0.0, hi - lo) / 1e3

rows = []
for e in hs:
    ov = sum(overlap_ms(e, n) for n in nccl)
    rows.append((e["dur"] / 1e3, ov, e["ts"]))
rows.sort()

fast = rows[:len(rows) // 2]
slow = rows[len(rows) // 2:]
print("%-10s %8s %12s %14s" % ("group", "ms", "nccl overlap", "overlap %"))
for label, grp in (("fast half", fast), ("slow half", slow)):
    ms = sum(r[0] for r in grp)
    ov = sum(r[1] for r in grp)
    print("%-10s %8.1f %12.1f %13.1f%%" % (label, ms, ov, 100 * ov / ms))

print("\nper-call: dur vs nccl overlap (sorted by duration)")
print("%8s %12s" % ("dur ms", "overlap ms"))
for dur, ov, ts in rows[::max(1, len(rows) // 12)]:
    print("%8.2f %12.2f" % (dur, ov))

# Is the excess explained? If dur = base + k*overlap, fit roughly.
import statistics
base = statistics.median(r[0] for r in fast)
excess = [(r[0] - base, r[1]) for r in slow]
num = sum(e * o for e, o in excess)
den = sum(o * o for _, o in excess)
k = num / den if den else 0
resid = [abs(e - k * o) for e, o in excess]
print("\nfast-half median %.2f ms; slow-half excess vs overlap: slope %.1fx, "
      "median residual %.2f ms" % (base, k, statistics.median(resid)))
print("=> overlap EXPLAINS the spread" if den and statistics.median(resid) < 1.0
      else "=> overlap does NOT explain the spread; look elsewhere")
