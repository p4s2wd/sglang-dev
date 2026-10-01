"""Is there a pipeline bubble, or was the '30% busy' figure an artifact?

Two numbers have been fighting. One pipeline-stage trace shows 1379 ms of device
time over 88 headshared-attention calls; the stage owns 11 of the 43 layers, so
that is 8 chunks and 172 ms of device work per 512-token chunk. Wall time per
chunk was inferred at ~538-600 ms, which gives 30% busy and a 3.4x pipeline-bubble
prize. But that wall figure came from log timestamps with one-second resolution
(gap histogram: 270 gaps of exactly 1000 ms, 247 of exactly 0 ms -- pure aliasing)
contaminated by idle time between probe requests.

The trace itself contains the answer and needs no logs: its own first and last
kernel timestamps bound the window it actually observed. If that window spans
8 x 538 = 4.3 s, the stage really is idle 70% of the time and the bubble is real.
If it spans ~1.5 s, then 1379 ms of kernels in a 1.5 s window means the stage is
~90% busy, there is no bubble, and prefill is simply at its kernel-speed limit --
which would also explain why concurrency, ring slack and a microbatch cap all
failed to move throughput.
"""
import gzip, glob, json, sys
from collections import defaultdict

d = sys.argv[1]
for f in sorted(glob.glob(d + "/*PP-3*EXTEND*")):
    ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    if not ks:
        continue
    first = min(e["ts"] for e in ks)
    last = max(e["ts"] + e["dur"] for e in ks)
    span = (last - first) / 1e6
    busy = sum(e["dur"] for e in ks) / 1e6
    hs = sum(1 for e in ks if "headshared" in e["name"] or "head_shared" in e["name"])
    nchunk = hs / 11.0
    print("%s" % f.split("/")[-1][:44])
    print("  kernels %d, device %.1f ms, window span %.2f s -> %.0f%% busy"
          % (len(ks), busy, span, 100 * busy / span))
    if hs:
        print("  headshared calls %d / 11 layers = %.1f chunks -> %.0f ms device work/chunk,"
              " %.0f ms wall/chunk" % (hs, nchunk, busy / nchunk * 1e3, span / nchunk * 1e3))
    # Largest gaps between consecutive kernels = where the idle actually is.
    ks.sort(key=lambda e: e["ts"])
    gaps = []
    cur_end, gstart = None, None
    for e in ks:
        if cur_end is not None and e["ts"] - cur_end > 0.02e6:
            gaps.append((e["ts"] - cur_end) / 1e6)
        cur_end = max(cur_end or 0, e["ts"] + e["dur"])
    gaps.sort(reverse=True)
    print("  inter-kernel gaps > 20 ms: %d, total %.0f ms, largest %s"
          % (len(gaps), sum(gaps) * 1e3, ["%.0f" % g for g in gaps[:5]]))
