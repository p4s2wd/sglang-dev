"""Rank decode's small-kernel overhead by DEVICE time per source line.

attrib_eager.py ranks by aten op; attrib_copy_lines.py ranks one op by host time.
This joins the whole chain -- python frame -> aten op -> cuda_runtime -> kernel --
so each kernel's device microseconds land on the sglang source line that issued
it, which is what a fusion roadmap must be sorted by.

Requires the eager trace (--disable-cuda-graph --disable-overlap-schedule):
graph replay collapses the op->kernel link and the overlap scheduler records the
op on a different thread than the kernel.
"""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
ev = json.load(gzip.open(path)).get("traceEvents", [])

# python frames per thread, for geometric nesting (cpu_op carries no Python id).
pyf = defaultdict(list)
for e in ev:
    if e.get("cat") == "python_function":
        pyf[e.get("tid")].append((e.get("ts", 0), e.get("dur", 0), e.get("name", "?")))
for tid in pyf:
    pyf[tid].sort(key=lambda t: t[0])

# cpu_op External id -> op name; cuda_runtime correlation -> External id.
ext2op = {}
for e in ev:
    if e.get("cat") == "cpu_op":
        eid = (e.get("args") or {}).get("External id")
        if eid is not None:
            ext2op[eid] = (e.get("name", "?"), e.get("ts", 0), e.get("tid"))

corr2ext = {}
for e in ev:
    if e.get("cat") == "cuda_runtime":
        a = e.get("args") or {}
        c = a.get("correlation")
        if c is not None:
            corr2ext[c] = a.get("External id")


def interesting(text):
    return "sglang" in text and ("srt/" in text or "kernels/" in text)


def innermost(tid, ts):
    best = None
    best_len = None
    for fts, fdur, name in pyf.get(tid, ()):
        if fts > ts:
            break
        if fts + fdur >= ts and interesting(name):
            if best_len is None or fdur < best_len:
                best, best_len = name, fdur
    return best


rows = defaultdict(lambda: [0, 0.0])
by_op = defaultdict(lambda: [0, 0.0])
unattributed = [0, 0.0]

for e in ev:
    if e.get("cat") != "kernel":
        continue
    c = (e.get("args") or {}).get("correlation")
    eid = corr2ext.get(c) if c is not None else None
    op = ext2op.get(eid)
    if op is None:
        unattributed[0] += 1
        unattributed[1] += e.get("dur", 0)
        continue
    name, ts, tid = op
    line = innermost(tid, ts) or f"(no frame) {name}"
    rows[line][0] += 1
    rows[line][1] += e.get("dur", 0)
    by_op[name][0] += 1
    by_op[name][1] += e.get("dur", 0)

tot = sum(v[1] for v in rows.values())
print(f"{tot/1e3:.2f}ms device attributed; "
      f"{unattributed[0]} kernels / {unattributed[1]/1e3:.2f}ms unattributed\n")

print("DEVICE time by source line:")
for line, (cnt, dur) in sorted(rows.items(), key=lambda kv: -kv[1][1])[:18]:
    print(f"  {dur/1e3:7.2f}ms x{cnt:<5d} avg {dur/cnt:6.1f}us  {line[:92]}")
