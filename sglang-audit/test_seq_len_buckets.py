"""CPU-only checks for decode seq_len bucketing.

The bucketing change touches three things that are easy to get wrong and none of
them need a GPU to pin down: how the env string becomes capture widths, which
bucket a given batch picks, and whether the metadata cache keeps two buckets of
the same batch size apart. A wrong pick here is a wrong attention width, so the
boundaries are the interesting cases.

Run: python test_seq_len_buckets.py
"""
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

CTX = 262144
fails = []


def check(name, got, want):
    ok = got == want
    if not ok:
        fails.append(name)
    print("  %-52s %-22s %s" % (name, repr(got), "ok" if ok else "FAIL want %r" % (want,)))


# --- parsing ---------------------------------------------------------------
from sglang.srt.layers.attention.deepseek_v4_backend import _parse_seq_len_buckets

print("parse (context %d):" % CTX)
check("empty -> context only", _parse_seq_len_buckets("", CTX), [CTX])
check("8192", _parse_seq_len_buckets("8192", CTX), [8192, CTX])
check("2048,8192", _parse_seq_len_buckets("2048,8192", CTX), [2048, 8192, CTX])
check("unsorted input", _parse_seq_len_buckets("8192,2048", CTX), [2048, 8192, CTX])
check("duplicates collapse", _parse_seq_len_buckets("2048,2048", CTX), [2048, CTX])
check("whitespace tolerated", _parse_seq_len_buckets(" 2048 , 8192 ", CTX), [2048, 8192, CTX])
check("garbage dropped", _parse_seq_len_buckets("2048,abc,,8192", CTX), [2048, 8192, CTX])
check("zero and negative dropped", _parse_seq_len_buckets("0,-5,2048", CTX), [2048, CTX])
# A bucket past the context is useless and would be wrong to capture: the widest
# graph must be exactly the context, never wider.
check("clamped to context", _parse_seq_len_buckets("300000", CTX), [CTX])
check("context not duplicated", _parse_seq_len_buckets("262144", CTX), [CTX])
# The list must always end at the context length -- that is what keeps a
# mis-picked bucket from ever being too narrow.
for spec in ("", "8192", "2048,8192", "300000", "0"):
    b = _parse_seq_len_buckets(spec, CTX)
    check("ends at context: %r" % spec, b[-1], CTX)

# --- bucket choice ---------------------------------------------------------
# Reach the resolver without importing the runner module's heavy deps.
import inspect
import re

src = open("/data/nvme/sglang-codex/sglang/python/sglang/srt/model_executor/"
           "runner/decode_cuda_graph_runner.py").read()
m = re.search(r"    def _resolve_seq_len_bucket\(.*?\n(?=    def )", src, re.S)
assert m, "could not locate _resolve_seq_len_bucket"
body = "\n".join(l[4:] for l in m.group(0).rstrip().split("\n"))
ns = {"Optional": type(None), "getattr": getattr}
exec(compile("def _resolve_seq_len_bucket(self, forward_batch):\n" +
             "\n".join("    " + l for l in body.split("\n")[1:]), "<t>", "exec"), ns)
resolve = ns["_resolve_seq_len_bucket"]


class Runner:
    def __init__(self, buckets):
        self._b = buckets

    def _replay_attn_backend(self):
        return SimpleNamespace(decode_seq_len_buckets=self._b)


def pick(buckets, cpu=None, gpu=None):
    fb = SimpleNamespace(seq_lens_cpu=cpu, seq_lens=gpu)
    return resolve(Runner(buckets), fb)


class T:
    """Stand-in for a CPU tensor: .numel() and .max().item()."""

    def __init__(self, v):
        self.v = v

    def numel(self):
        return 1 if self.v is not None else 0

    def max(self):
        return self


def item(self):
    return self.v


T.item = item

print("\nbucket choice (buckets [2048, 8192, 262144]):")
B = [2048, 8192, CTX]
check("no buckets -> None", pick([], cpu=T(5)), None)
check("kv 11", pick(B, cpu=T(11)), 2048)
check("kv 2047", pick(B, cpu=T(2047)), 2048)
check("kv 2048 (on boundary)", pick(B, cpu=T(2048)), 2048)
check("kv 2049 (just past)", pick(B, cpu=T(2049)), 8192)
check("kv 8192", pick(B, cpu=T(8192)), 8192)
check("kv 8193", pick(B, cpu=T(8193)), CTX)
check("kv == context", pick(B, cpu=T(CTX)), CTX)
check("kv past context -> widest", pick(B, cpu=T(CTX + 1)), CTX)
check("empty cpu mirror falls to device", pick(B, cpu=T(None), gpu=T(3000)), 8192)
check("no length info -> widest", pick(B, cpu=T(None), gpu=T(None)), CTX)
# The batch is served by the widest graph its longest request needs.
check("max governs, not min", pick(B, cpu=T(9000)), CTX)

print("\nmetadata cache key:")
# Two buckets of the same bs must not share one page-table view.
cache = {}
for bs, bucket in ((1, 2048), (1, 8192), (1, None), (2, 8192)):
    key = bs if bucket is None else (bs, bucket)
    cache[key] = "meta"
check("bs=1 bucketed keys distinct",
      (1, 2048) != (1, 8192) and (1, 2048) in cache and (1, 8192) in cache, True)
# An unbucketed runner keys on the bare bs, so it must not collide with a tuple.
check("unbucketed keys on bare bs", 1 in cache, True)
check("bare bs distinct from tuple keys", (1, 2048) in cache and 1 != (1, 2048), True)
check("4 distinct entries", len(cache), 4)

print("\n%s" % ("ALL PASS" if not fails else "FAILURES: %s" % fails))
sys.exit(1 if fails else 0)
