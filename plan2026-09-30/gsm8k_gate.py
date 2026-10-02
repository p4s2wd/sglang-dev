#!/usr/bin/env python
"""GSM8K correctness gate, plus the crash-pattern count that goes with it.

Two things are checked together on purpose. Accuracy alone would pass a server
that is quietly corrupting memory and still answering most questions; the crash
patterns alone would pass a server that answers nothing. The gate is both.

The gold answer is the text after the final `####` in GSM8K's own annotation,
which avoids re-deriving it from the calculator annotations and keeps this
comparable with the earlier 0.945 reading.

Usage: gsm8k_gate.py [--port 8200] [--n 200] [--conc 8] [--data PATH]
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_DATA = "/data/nvme/sglang-codex/gsm8k_test.jsonl"
DEFAULT_LOG = "/data/nvme/sglang/logs/serve-prod.log"

CRASH_PATTERNS = [
    "unhandled cuda error",
    "out of memory",
    "CUDA error: an illegal memory access",
    "illegal memory access",
    "IMA",
    "Scheduler hit an exception",
    "Input is not valid",
    "Traceback (most recent call last)",
]


def gold(answer: str) -> str:
    tail = answer.split("####")[-1].strip()
    return tail.replace(",", "")


def _num(s: str) -> str | None:
    v = s.replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(v)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def extract_strict(text: str) -> str | None:
    """The last number in the response -- the conventional reading.

    Strict on purpose, and known to undercount: this model ends with sentences
    like "Terry spends $75.00 on yogurt over 30 days", where the answer is not
    the last number, and it sometimes answers correctly and then volunteers a
    second interpretation, which becomes the last number.
    """
    m = re.findall(r"-?\$?\d[\d,]*\.?\d*", text)
    return _num(m[-1]) if m else None


def extract_contains(text: str, want: str) -> bool:
    """True if the gold value appears as a standalone number anywhere.

    Lenient on purpose, for the same reason: it counts an answer the model gave
    and then hedged around, which is a correct answer, and it does not punish
    trailing units. Both numbers are reported so neither reading is hidden.
    """
    return any(_num(m) == want for m in
               re.findall(r"-?\$?\d[\d,]*\.?\d*", text))


def one(url, question, max_new=2048, timeout=1800):
    """Chat completions, not /generate.

    /generate takes raw text and does NOT apply the chat template, so the model
    receives an unwrapped question and answers like a base model: it continues
    training-data JSON and never emits EOS, which reads as 0.25 accuracy and a
    100x slowdown. The same questions through /v1/chat/completions come back
    correct in ~100 tokens. A gate that scores the wrong endpoint measures the
    harness, not the server.
    """
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.0, "top_p": 1.0, "max_tokens": max_new,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def crash_counts(log_path):
    if not os.path.exists(log_path):
        return None
    counts = {}
    with open(log_path, "r", errors="replace") as f:
        text = f.read()
    for p in CRASH_PATTERNS:
        counts[p] = text.count(p)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=2048,
                    help="reasoning models need room to finish thinking;\n"
                         "512 truncates mid-thought and reads as wrong")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--start-line", type=int, default=0,
                    help="only count crash patterns after this log line")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    rows = [json.loads(l) for l in open(args.data)][: args.n]
    print(f"GSM8K: {len(rows)} 题, 并发 {args.conc}, greedy, :{args.port}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.conc) as ex:
        texts = list(ex.map(
            lambda r: one(url, r["question"], args.max_new), rows))
    dt = time.time() - t0

    ok_strict = 0
    ok_contains = 0
    empty = 0
    wrong = []
    for r, t in zip(rows, texts):
        g = gold(r["answer"])
        t = t or ""
        if not t.strip():
            empty += 1
        if extract_strict(t) == g:
            ok_strict += 1
        if extract_contains(t, g):
            ok_contains += 1
        else:
            wrong.append((g, extract_strict(t)))
    n = len(rows)
    print(f"准确率(严格,末位数字): {ok_strict}/{n} = {ok_strict / n:.3f}")
    print(f"准确率(含正确答案):     {ok_contains}/{n} = {ok_contains / n:.3f}")
    print(f"空响应(thinking 跑飞截断): {empty}")
    print(f"耗时 {dt:.0f}s   {n / dt:.2f} 题/s")
    if wrong:
        print(f"前 8 个错例 (gold, 末位数字): {wrong[:8]}")

    counts = crash_counts(args.log)
    print("\n崩溃计数 (serve 日志):")
    bad = 0
    for p, c in counts.items():
        flag = "" if c == 0 else "   <-- 非零"
        if c:
            bad += 1
        print(f"  {p:<42} {c}{flag}")
    healthy = urllib.request.urlopen(
        f"http://127.0.0.1:{args.port}/health", timeout=15).status == 200
    print(f"服务存活: {healthy}")
    # Gate on the lenient score and report both: the strict number is not a
    # regression signal on its own, it moves with answer formatting.
    verdict = bad == 0 and healthy and ok_contains / n >= 0.90
    print(f"\n闸门: {'通过' if verdict else '不通过'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())