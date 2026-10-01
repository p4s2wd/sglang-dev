"""Identify which aten op drives the fp16 gemvx, from recorded operand shapes.

The DECODE trace has record_shapes=True, so each cpu_op carries its operand
sizes. The device-side kernel name (gemvx<__half>) does not say which weight it
is, but the operand shapes do: a [8192, 4096] fp16 operand is wo_a, a
[4096, ...] pair is something else. Prints each distinct (op, shapes) with its
call count and the bytes the weight operand would move per call.
"""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
data = json.load(gzip.open(path))
events = data.get("traceEvents", data if isinstance(data, list) else [])

rows = defaultdict(lambda: [0, 0.0])
for e in events:
    if e.get("ph") != "X" or e.get("cat") != "cpu_op":
        continue
    a = e.get("args") or {}
    shapes = a.get("InputShapes") or a.get("input_shapes")
    if not shapes:
        continue
    # Only ops whose largest 2-D operand looks like a weight (both dims > 512).
    big = None
    for s in shapes:
        if isinstance(s, list) and len(s) == 2 and min(s) > 512:
            if big is None or (s[0] * s[1]) > (big[0] * big[1]):
                big = s
    if big is None:
        continue
    key = (e.get("name", "?"), tuple(big))
    rows[key][0] += 1
    rows[key][1] += e.get("dur", 0)

print(f"{len(rows)} distinct (op, weight-shape) pairs")
for (name, shape), (cnt, dur) in sorted(
        rows.items(), key=lambda kv: -kv[1][0] * kv[1][1])[:18]:
    nbytes = shape[0] * shape[1] * 2
    print(f"  x{cnt:<5d} host {dur/1e3:7.2f}ms  {str(shape):18s} "
          f"{nbytes/1e6:6.1f} MiB(fp16)  {name[:52]}")
