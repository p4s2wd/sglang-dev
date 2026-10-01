"""Production W4A16 rate: bytes the kernel must read, divided by its measured time.

The isolated benchmark says the repacked v3 kernel runs at 137 GB/s, and route (2)
targets ~500 GB/s. Judge it on production numbers instead: per stage per token the
routed experts are 11 layers x topk 6 x 13.37 MB / TP2 = 0.441 GB, and the trace gives
the kernel's duration and call count, so the achieved rate follows.

Also report occupancy and the launch geometry, since grid [64,6,1] with 32-thread
blocks and 18% achieved occupancy is the signature of a launch too small to saturate
the memory system, which would explain a sub-peak rate without the kernel itself being
inefficient.
"""
import gzip, glob, json

PER_EXPERT = 13.37e6      # bytes incl. scales, from bytes_truth2.py
LAYERS = 11
TOPK = 6
TP = 2
BW = 616e9

for d in ("profiles/exact-1790060350",):
    for pp in range(4):
        f = max(glob.glob(d + "/*TP-0-PP-%d.trace.json.gz" % pp))
        ev = json.load(gzip.open(f))["traceEvents"]
        ks = [e for e in ev if e.get("cat") == "kernel"]
        steps = max(1, sum(1 for e in ks if "hc_split_sinkhorn" in e["name"]) / 22.0)
        ex = [e for e in ks if "w4a16" in e["name"]]
        if not ex:
            continue
        calls = len(ex)
        ms = sum(e["dur"] for e in ex) / 1e3
        # bytes actually read per step: each MoE layer reads topk experts, split over
        # the calls that layer makes (gate/up and down are separate calls)
        bytes_step = LAYERS * TOPK * PER_EXPERT / TP
        rate = bytes_step / (ms / steps * 1e-3) / 1e9
        occ = [e["args"].get("est. achieved occupancy %") for e in ex[:50]]
        grid = ex[0]["args"].get("grid")
        print("PP%d  %4d calls / %4.1f steps = %5.1f calls/step  %.4f ms/call  "
              "%.2f ms/token  -> %5.0f GB/s (%.0f%% of 616)  occ %s%% grid %s"
              % (pp, calls, steps, calls / steps, ms / calls, ms / steps, rate,
                 100 * rate / 616, int(sum(occ) / len(occ)), grid))
        print("      routed bytes/step %.1f MB; kernel time covers %.0f%% of the "
              "stage's routed share" % (bytes_step / 1e6, 100 * (ms / steps) /
                                        (bytes_step / BW * 1e3)))
