"""Which tensor shapes drive decode's elementwise time?

record_shapes=True puts the operand sizes on each cpu_op, so instead of guessing
why a batch-1 decode spends 42% of device time on elementwise work we can read
off the shapes. Prints the largest operands by bytes-touched-per-call.
"""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
data = json.load(gzip.open(path))
events = data.get("traceEvents", data if isinstance(data, list) else [])

# args.input_shapes / input_dtypes are recorded per cpu_op.
rows = defaultdict(lambda: [0, 0.0, None])
for e in events:
    if e.get("ph") != "X" or e.get("cat") != "cpu_op":
        continue
    a = e.get("args") or {}
    shapes = a.get("input_shapes")
    if not shapes:
        continue
    dtypes = a.get("input_dtypes") or []
    sizes = {"f32": 4, "float32": 4, "f16": 2, "half": 2, "bf16": 2,
             "f64": 8, "i32": 4, "i64": 8, "i8": 1, "u8": 1, "bool": 1}

    def nbytes(shape, dt):
        if not shape:
            return 0
        n = 1
        for d in shape:
            n *= int(d)
        return n * sizes.get(str(dt).lower().replace("torch.", ""), 2)

    total = 0
    if isinstance(shapes, str):
        continue
    for i, sh in enumerate(shapes):
        dt = dtypes[i] if i < len(dtypes) else "f16"
        total += nbytes(sh, dt)
    key = (e["name"], json.dumps(shapes)[:110])
    r = rows[key]
    r[0] += 1
    r[1] += e.get("dur", 0)
    r[2] = total

print(f"{'us/call':>8} {'bytes/call':>12} {'count':>6}  op / shapes")
ranked = sorted(rows.items(), key=lambda kv: -(kv[1][2] or 0))
for (name, shapes), (cnt, dur, nb) in ranked[:22]:
    print(f"{dur/cnt:>8.1f} {nb:>12,} {cnt:>6}  {name} {shapes}")
