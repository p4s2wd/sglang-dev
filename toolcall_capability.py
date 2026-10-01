#!/usr/bin/env python
"""Is the model capable of tool calling at all, and of not repeating a call?

The round trip is lossless, so history corruption is off the table. But "the serving
layer is faithful" is not the same as "the model is fine", and the failure could still
be the model's. The question that separates them:

    Given a conversation where a command's output is ALREADY present, does the model
    re-issue that command?

If it handles that correctly at short context but fails at 64K, the defect is
long-context degradation -- the model loses track of what it already did. If it fails
even at short context, the checkpoint cannot do this at all, and no serving change
will help. These need very different responses, so they are measured separately.

Each case is run more than once, because this investigation already produced two
opposite conclusions from single runs.
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8200/v1/chat/completions"
TOOLS = [
    {"type": "function", "function": {"name": "bash", "description":
        "Run a shell command.", "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read", "description":
        "Read a file.", "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
]
GREP = ('cd /data/nvme/sglang-codex && grep -rn "decode\\|tok/s\\|13.55" '
        'sglang-sm75-progress.md | head -20')
GREP_OUT = "39:| B4 | DSA Triton FP16 (done) | 100% | decode 13.55 tok/s"

fails = 0


def ask(messages, extra=None, max_tokens=800, timeout=900):
    body = {"model": "deepseek-v4-flash", "messages": messages,
            "max_tokens": max_tokens, "stream": False, "tools": TOOLS}
    body.update(extra or {"temperature": 0.0})
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    ch = r["choices"][0]
    return ch, r.get("usage", {}), round(time.time() - t0)


def call_of(ch):
    tc = ch["message"].get("tool_calls") or []
    if not tc:
        return None
    try:
        return json.loads(tc[0]["function"]["arguments"]).get("command", "")
    except Exception:
        return tc[0]["function"]["arguments"][:80]


def check(name, ok, detail=""):
    global fails
    if not ok:
        fails += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")


def turn(cmd, out, nxt):
    return [{"role": "user", "content": "Summarise the decode numbers in the progress doc."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": out},
            {"role": "user", "content": nxt}]


print("=== A. 基础能力:工具调用是否正常(短上下文)===")
ch, us, sec = ask([{"role": "user", "content": "List the .md files in /tmp using bash."}])
c = call_of(ch)
check("能正确发起工具调用", c is not None and "ls" in (c or ""), f"cmd={(c or '')[:70]!r}")

print("\n=== B. 关键:输出已经在上下文里,会不会重跑同一条命令(短上下文)===")
for i in range(3):
    ch, us, sec = ask(turn(GREP, GREP_OUT, "Now summarise those numbers for me."))
    c = call_of(ch)
    repeated = c is not None and "sglang-sm75-progress.md" in c and "grep" in c
    check(f"run{i}: 没有重跑已有输出的命令", not repeated,
          f"cmd={(c or ch['message'].get('content') or '')[:80]!r}")

print("\n=== C. 同一结果在历史里出现 6 次后,会不会重跑 ===")
msgs = [{"role": "user", "content": "Summarise the decode numbers in the progress doc."}]
for k in range(6):
    msgs += [{"role": "assistant", "content": None, "tool_calls": [
        {"id": f"c{k}", "type": "function",
         "function": {"name": "bash", "arguments": json.dumps({"command": GREP})}}]},
        {"role": "tool", "tool_call_id": f"c{k}", "content": GREP_OUT}]
msgs += [{"role": "user", "content": "Now summarise those numbers for me."}]
ch, us, sec = ask(msgs)
c = call_of(ch)
repeated = c is not None and "sglang-sm75-progress.md" in c and "grep" in c
check("6 次重复结果后仍不重跑", not repeated,
      f"in={us.get('prompt_tokens')} cmd={(c or ch['message'].get('content') or '')[:80]!r}")

print("\n=== D. 同一结果在历史里出现 20 次后(短上下文,纯体积) ===")
msgs = [{"role": "user", "content": "Summarise the decode numbers in the progress doc."}]
for k in range(20):
    msgs += [{"role": "assistant", "content": None, "tool_calls": [
        {"id": f"c{k}", "type": "function",
         "function": {"name": "bash", "arguments": json.dumps({"command": GREP})}}]},
        {"role": "tool", "tool_call_id": f"c{k}", "content": GREP_OUT}]
msgs += [{"role": "user", "content": "Now summarise those numbers for me."}]
for i in range(2):
    ch, us, sec = ask(msgs)
    c = call_of(ch)
    repeated = c is not None and "sglang-sm75-progress.md" in c and "grep" in c
    check(f"run{i}: 20 次重复结果后不重跑", not repeated,
          f"in={us.get('prompt_tokens')} cmd={(c or ch['message'].get('content') or '')[:80]!r}")

print(f"\nFAILS = {fails}")
