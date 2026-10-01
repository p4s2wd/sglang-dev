"""Which source line issues decode's 33 dtype copies per layer?

attrib_eager.py shows aten::copy_ is the single largest small-kernel source
(1310 launches / 6.64 ms over 40 layer-steps, 34% of all small-kernel time).
Copies are pure overhead at batch 1: they move a few KB to satisfy a dtype or
contiguity requirement, and each pays the 2 us device-side floor.

with_stack=True records Python frames; a cpu_op carries "Python id" and
"Python parent id", and python_function events carry the same ids. Walk up from
each copy_ to the deepest frame inside sglang's model/layer code -- that is the
line to change.
"""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
TARGET = sys.argv[2] if len(sys.argv) > 2 else "aten::copy_"
ev = json.load(gzip.open(path)).get("traceEvents", [])

# cpu_op events do NOT carry a Python id in this trace format, so the stack has
# to be recovered geometrically: python_function events are duration events on a
# thread, and the frames enclosing a cpu_op are exactly those on the same tid
# whose [ts, ts+dur] window contains the op's ts. Deepest = shortest enclosing
# window.
pyf = defaultdict(list)
for e in ev:
    if e.get("cat") == "python_function":
        pyf[e.get("tid")].append(e)
for tid in pyf:
    pyf[tid].sort(key=lambda e: e.get("ts", 0))


def interesting(text):
    return "sglang" in text and ("srt/" in text or "kernels/" in text)


def enclosing(tid, ts):
    out = []
    for f in pyf.get(tid, ()):
        fts = f.get("ts", 0)
        if fts > ts:
            break
        if fts + f.get("dur", 0) >= ts:
            out.append(f.get("name", "?"))
    # shortest enclosing window last = innermost frame
    out.sort(key=lambda n: 0)
    return out


rows = defaultdict(lambda: [0, 0.0])
for e in ev:
    if e.get("cat") != "cpu_op" or e.get("name") != TARGET:
        continue
    ts = e.get("ts", 0)
    ch = enclosing(e.get("tid"), ts)
    line = next((f for f in reversed(ch) if interesting(f)), None)
    if line is None:
        line = ch[-1] if ch else "(no python frame)"
    rows[line][0] += 1
    rows[line][1] += e.get("dur", 0)

tot = sum(v[1] for v in rows.values())
n = sum(v[0] for v in rows.values())
print(f"{TARGET}: {n} launches, host {tot/1e3:.2f}ms over this trace")
print(f"{len(rows)} distinct source lines\n")
for line, (c, d) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:14]:
    print(f"  x{c:<5d} host {d/1e3:6.2f}ms  {line[:104]}")
