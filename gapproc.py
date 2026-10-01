"""Where are the idle gaps: between chunks, or at the window edges?

The PP-3 EXTEND trace spans 4.65 s with 1.379 s of device work (30% busy) and
three gaps over 20 ms totalling 3.2 s. That 30% has been read as a pipeline
bubble worth 3.4x. But the profiler window also covers the time it spends waiting
for the next probe request to arrive, and idle there is not a bubble -- it is an
empty queue.

Decisive test: locate each gap relative to the chunk boundaries. Chunk starts are
identifiable because each chunk runs the stage's 11 headshared-attention calls
back to back. A gap sitting between two chunks' attention bursts is a real stall
in the middle of prefill; a gap before the first burst or after the last is just
the profiler waiting. If the 3.2 s is all at the edges, the stage is nearly busy
whenever there is work and prefill is kernel-limited, not schedule-limited.
"""
import gzip, glob, json, sys

d = sys.argv[1]
f = max(glob.glob(d + "/*PP-3*EXTEND*"))
ev = json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]
ks = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
t0 = ks[0]["ts"]

# Chunk boundaries from the attention calls: 11 per chunk, so group them.
hs = [e for e in ks if "headshared" in e["name"] or "head_shared" in e["name"]]
bounds = []
for i in range(0, len(hs), 11):
    grp = hs[i:i + 11]
    if len(grp) == 11:
        bounds.append((grp[0]["ts"], grp[-1]["ts"] + grp[-1]["dur"]))
print("%d chunks identified" % len(bounds))

# Walk the timeline, recording gaps and which chunk each side belongs to.
cur_end = ks[0]["ts"] + ks[0]["dur"]
cur_idx = 0
for e in ks[1:]:
    gap = e["ts"] - cur_end
    if gap > 20e3:
        before = sum(1 for a, b in bounds if b <= cur_end)
        after = sum(1 for a, b in bounds if a >= e["ts"])
        pos = ("BEFORE all chunks (waiting for work)" if before == 0 else
               "AFTER all chunks (queue empty)" if before >= len(bounds) else
               "BETWEEN chunk %d and chunk %d (REAL STALL)" % (before, len(bounds) - after))
        print("  gap %7.0f ms at t=%6.0f ms  %s" % (gap / 1e3, (cur_end - t0) / 1e3, pos))
    cur_end = max(cur_end, e["ts"] + e["dur"])

# Busy fraction restricted to the span that actually contains chunks.
if bounds:
    a, b = bounds[0][0], bounds[-1][1]
    inside = [e for e in ks if e["ts"] >= a and e["ts"] + e["dur"] <= b]
    span = (b - a) / 1e6
    busy = sum(e["dur"] for e in inside) / 1e6
    print("\nactive span (first chunk start -> last chunk end): %.2f s" % span)
    print("device work inside it: %.0f ms -> %.0f%% busy" % (busy * 1e3, 100 * busy / span))
    print("chunks %d -> %.0f ms wall/chunk, %.0f ms device/chunk"
          % (len(bounds), span / len(bounds) * 1e3, busy / len(bounds) * 1e3))
    print("throughput implied by active span: %.0f tok/s at 512 tok/chunk"
          % (512 * len(bounds) / span))
