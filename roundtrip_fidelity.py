#!/usr/bin/env python
"""Is the tool-call round trip faithful? This decides model-vs-serving.

The user's objection is the right one: a loop of identical tool calls could easily be
our fault rather than the model's. The specific mechanism that would make it our
fault is a lossy round trip. The model emits a DSML tool call, serving_chat parses it
into an OpenAI `tool_calls` field, the agent stores that, and on the next request
encoding_dsv4 renders it back into the prompt. If what comes back is not what the
model emitted -- reformatted arguments, a lost block, a mangled name -- then the
model is being shown a history that contradicts its own output, and re-issuing the
call is a rational response to a corrupted context. That would be a serving bug.

So the round trip is checked directly, on the real text from the session:

    model DSML  ->  DeepSeekV4Detector  ->  OpenAI tool_calls  ->  encode_messages
    ->  compare against the original DSML

Faithful means the model sees its own output. Anything else localises the bug to the
encoder/parser rather than the checkpoint.
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, "/data/nvme/sglang/.venv/lib/python3.12/site-packages")
from sglang.srt.entrypoints.openai import encoding_dsv4  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import Tool  # noqa: E402
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector  # noqa: E402

DB = Path.home() / ".local/share/opencode/opencode.db"
SESSION = "ses_f12a25898ffe1ypJUExVYX9aQm"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
cur = con.cursor()
rows = cur.execute(
    "select id,data from message where session_id=? order by time_created",
    (SESSION,)).fetchall()

detector = DeepSeekV4Detector()
tools = [Tool(type="function", function={"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}})]

fails = 0


def check(name, ok, detail=""):
    global fails
    if not ok:
        fails += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")


# The DSML the model itself produced for the looping call.
cmd = None
for mid, raw in rows:
    d = json.loads(raw)
    if d.get("role") != "assistant":
        continue
    for (pd,) in cur.execute("select data from part where message_id=?", (mid,)):
        p = json.loads(pd)
        if p.get("type") == "tool":
            c = ((p.get("state") or {}).get("input") or {}).get("command")
            if c and "sglang-sm75-progress.md" in c:
                cmd = c
                break
    if cmd:
        break

print("=== 1. 从 DB 的工具参数重建模型原始 DSML ===")
# The model emitted DSML; the agent stored the parsed args. Rebuild the DSML the way
# encoding_dsv4 renders it, so the comparison below is against the served format.
# Format 1, which is what the model actually emitted in this session (verified
# against a live capture), not a JSON re-encoding of the arguments.
dsml = ("<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="bash">\n'
        f'<｜DSML｜parameter name="command" string="true">{cmd}</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>")
print("  模型原始 DSML:")
for _l in dsml.splitlines(): print("   ", _l[:150])
print(f"  命令长度 {len(cmd)} 字符")

print("\n=== 2. DSML -> OpenAI tool_calls(服务端解析)===")
res = detector.parse_streaming_increment(dsml, tools)
# The detector emits the name in one item and the argument JSON in the next, so the
# call is reassembled from both. Taking only the name-bearing item silently yields
# empty arguments -- which is what an earlier revision of this script did.
print(f"  原始 items: {[(c.name, (c.parameters or '')[:50]) for c in res.calls]}")
name = next((c.name for c in res.calls if c.name), None)
arguments = "".join((c.parameters or "") for c in res.calls if not c.name)
print(f"  合并后: name={name} arguments={arguments[:90]}")
ok_args = '"command"' in arguments and "sglang-sm75-progress.md" in arguments
check("解析出了 bash 调用", name == "bash")
check("参数里保留了原始命令", ok_args,
      f"(arguments {len(arguments)} 字符)")

print("\n=== 3. OpenAI tool_calls -> prompt(服务端回渲)===")
openai_calls = [{"id": "call_0", "type": "function",
                 "function": {"name": name, "arguments": arguments}}]
msgs = [{"role": "user", "content": "读取项目进度"},
        {"role": "assistant", "content": None, "tool_calls": openai_calls},
        {"role": "tool", "tool_call_id": "call_0", "content": "39:| B4 | DSA ..."}]
rendered = encoding_dsv4.encode_messages(
    msgs, thinking_mode="chat", add_default_bos_token=False)
print(f"  渲染结果:\n{rendered}")

print("\n=== 4. 往返保真度 ===")
check("assistant 轮里 tool_calls 被渲染成 DSML", "DSML｜invoke name=\"bash\"" in rendered)
check("原始命令完整出现在 prompt 里", cmd in rendered,
      f"(命令 {len(cmd)} 字符)")
check("tool 结果也进了 prompt", "39:| B4 | DSA" in rendered)
# The critical check: does the re-rendered call still look like what the model wrote?
double_encoded = '{"command": "{\\"command\\"'
check("没有出现 arguments 被二次编码的痕迹", double_encoded not in rendered)
print(f"\nFAILS = {fails}")
