"""Are the model's norms fused or eager in the running server?

The fused flashinfer norm is 3.3-9.3x faster than the eager x.pow(2).mean(-1)
equivalent at production shapes, and the profile shows a pow_tensor_scalar kernel
at 126 us/call -- the signature of eager. But the server log has no JIT failure, so
the norms may already be fused and the pow may come from somewhere else entirely.

Decidable from the existing trace: the fused path launches a kernel whose name
contains rmsnorm; the eager path launches pow_tensor_scalar + reduce_kernel(mean) +
muls. Count both families and their per-layer rate. 43 layers / 4 PP stages = 11
layers per stage, each with two norms, so a fused norm should appear ~11 times per
chunk and an eager one ~22-33 (pow, mean, mul).
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
for f in sorted(glob.glob(d + "/*PP-3*EXTEND*"))[:1]:
    ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    fam = defaultdict(lambda: [0.0, 0])
    for e in ks:
        n = e["name"]
        low = n.lower()
        if "rmsnorm" in low or "rms_norm" in low:
            fam["FUSED rmsnorm"][0] += e["dur"] / 1e3; fam["FUSED rmsnorm"][1] += 1
        if "pow_tensor_scalar" in low:
            fam["eager pow(2)"][0] += e["dur"] / 1e3; fam["eager pow(2)"][1] += 1
        if "reduce_kernel" in low and "MeanOps" in n:
            fam["eager mean"][0] += e["dur"] / 1e3; fam["eager mean"][1] += 1
        if "rsqrt" in low:
            fam["rsqrt"][0] += e["dur"] / 1e3; fam["rsqrt"][1] += 1
    hs = sum(1 for e in ks if "headshared" in e["name"])
    nchunk = max(1, hs / 11.0)
    print("%s: %d kernels, %d headshared -> %.1f chunks" % (f.split("/")[-1][:40], len(ks), hs, nchunk))
    for k, (ms, n) in sorted(fam.items(), key=lambda x: -x[1][0]):
        print("  %-16s %7.1f ms  %5d calls  %6.1f per chunk  %5.1f us/call"
              % (k, ms, n, n / nchunk, ms * 1e3 / n))
    if not fam:
        print("  no norm-family kernels found at all")
