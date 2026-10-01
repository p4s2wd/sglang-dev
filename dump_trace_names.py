"""Dump raw kernel names from a trace, untruncated, to identify what the
analyzer's trimmed names actually are."""
import gzip
import json
import sys
from collections import Counter

path = sys.argv[1]
cat = sys.argv[2] if len(sys.argv) > 2 else "kernel"
top = int(sys.argv[3]) if len(sys.argv) > 3 else 10

data = json.load(gzip.open(path))
events = data.get("traceEvents", data if isinstance(data, list) else [])
tot, cnt = Counter(), Counter()
for e in events:
    if e.get("ph") != "X" or e.get("cat") != cat:
        continue
    tot[e["name"]] += e.get("dur", 0)
    cnt[e["name"]] += 1

acc = sum(tot.values()) or 1
print(f"{cat}: {acc/1000:.2f} ms total, {sum(cnt.values())} events")
for name, t in tot.most_common(top):
    print(f"  {100*t/acc:5.1f}% {t/1000:8.2f}ms x{cnt[name]:<6} {name[:170]}")
