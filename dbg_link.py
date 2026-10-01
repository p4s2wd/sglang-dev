"""Figure out how kernels link to CPU ops in this profiler's output."""
import gzip, glob, json, sys
from collections import Counter

f = max(glob.glob(sys.argv[1] + "/*DECODE*.trace.json.gz"))
ev = json.load(gzip.open(f))["traceEvents"]
ks = [e for e in ev if e.get("cat") == "kernel"]
rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
print("file", f.split("/")[-1], "kernels", len(ks), "runtime", len(rt))
print("kernel args keys:", Counter(tuple(sorted(e.get("args", {}).keys())) for e in ks).most_common(3))
print("runtime names:", Counter(e["name"] for e in rt).most_common(4))
print("runtime args keys:", Counter(tuple(sorted(e.get("args", {}).keys())) for e in rt).most_common(3))
print("sample kernel:", json.dumps(ks[500])[:300])
print("sample runtime:", json.dumps(rt[50])[:300])
cpu = [e for e in ev if e.get("cat") == "cpu_op"]
print("cpu_op count:", len(cpu), "sample:", json.dumps(cpu[200])[:300] if len(cpu) > 200 else "-")
