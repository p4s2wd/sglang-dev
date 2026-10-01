"""Attribute decode's elementwise kernels to the model source lines that launch them.

Inside a CUDA graph the op->kernel correlation collapses (everything joins to
cudaGraphLaunch), so the only way to find out what launches ~93 elementwise
kernels per layer is the Python stack recorded with with_stack=True. Kernel
events carry "Python id"; the stack frames are separate events keyed by that id,
innermost last. Reports the deepest frame that is sglang model/layer code, which
is the line a fusion pass would rewrite.
"""
import gzip
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
ev = json.load(gzip.open(path)).get("traceEvents", [])

# Python frames: ph "p" (parent) / "c" (child) events with cat python_function,
# or the "Python id" argument on kernel events pointing at a frame chain.
frames = {}
for e in ev:
    if e.get("cat") == "python_function":
        frames[e.get("id", e.get("pid"))] = e

def is_sglang(text):
    return "sglang" in text and "site-packages" not in text


def classify(name):
    n = re.sub(r"\s*\(.*", "", name)
    n = re.sub(r"<.*", "", n).strip()
    if "elementwise" in n or "vectorized_elementwise" in n:
        return "elementwise"
    if "reduce_kernel" in n or "reduce_1Block" in n:
        return "reduce"
    if "index_elementwise" in n or "index_kernel" in n:
        return "index"
    return None


# Group by the python frame chain recorded on the kernel event itself.
rows = defaultdict(lambda: [0, 0.0])
unattributed = [0, 0.0]
for e in ev:
    if e.get("cat") != "kernel":
        continue
    kind = classify(e.get("name", ""))
    if kind is None:
        continue
    a = e.get("args") or {}
    stack = a.get("Python id") or a.get("python_id") or a.get("Stack")
    line = None
    if isinstance(stack, list):
        for fr in reversed(stack):
            s = str(fr)
            if is_sglang(s):
                line = s
                break
        if line is None and stack:
            line = str(stack[-1])
    if line is None:
        unattributed[0] += 1
        unattributed[1] += e.get("dur", 0)
        line = "(no stack recorded)"
    rows[(kind, line)][0] += 1
    rows[(kind, line)][1] += e.get("dur", 0)

tot = sum(v[1] for v in rows.values())
print(f"{tot/1e3:.2f}ms of elementwise/reduce/index kernels")
print(f"unattributed: {unattributed[0]} launches, {unattributed[1]/1e3:.2f}ms\n")
for (kind, line), (c, d) in sorted(rows.items(), key=lambda kv: -kv[1][1])[:20]:
    print(f"  {d/1e3:7.2f}ms x{c:<5d} {kind:12s} {line[:96]}")
