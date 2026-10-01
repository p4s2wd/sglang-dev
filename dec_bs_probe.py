"""What batch size does production decode actually run at?

The gate change to SGLANG_SM75_HEADSHARED_MIN_BATCH=2 should have moved decode from
the per-head kernel to the head-shared one at bs>=2. Post-change DECODE traces still
show only the per-head kernel, so either decode never reaches bs=2, or the change is
not reaching the dispatch. The server logs the answer directly: every decode batch
line carries #running-req. Two concurrent HTTP requests do not guarantee a shared
decode step -- if they never overlap, every step is bs=1, where per-head is the
correct kernel and the gate change is a no-op in production.
"""
import subprocess, re, sys
log = sys.argv[1]
txt = subprocess.run(["grep", "-aE", "Decode batch", log], capture_output=True, text=True).stdout
pat = re.compile(r"Decode batch.*?#running-req: (\d+).*?#batch-size: (\d+)|Decode batch.*?#running-req: (\d+)")
from collections import Counter
c = Counter()
for m in pat.finditer(txt):
    c[m.group(2) or m.group(3)] += 1
print("decode batch lines: %d" % len(txt.splitlines()))
print("#running-req histogram:", dict(c.most_common(8)))
