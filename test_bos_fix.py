#!/usr/bin/env python
"""Verify the BOS-stripping fix without needing a server restart.

Imports the patched serving_chat module and calls the helper directly, then feeds
the result through the real dsv4 encoder to confirm no BOS survives into the
prompt. Also checks the two things that must NOT change: roles/tool calls are
untouched, and the caller's list is not mutated.
"""
import sys

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

BOS = "<｜begin▁of▁sentence｜>"

from sglang.srt.entrypoints.openai import encoding_dsv4
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

strip = OpenAIServingChat._strip_bos_from_messages

fails = 0


def check(name, cond, detail=""):
    global fails
    if not cond:
        fails += 1
    print(f"  {'OK  ' if cond else 'FAIL'} {name}{('  ' + detail) if detail else ''}")


print("1. single BOS in user content")
m = [{"role": "user", "content": BOS + "Say OK."}]
out = strip(m)
check("BOS removed", BOS not in out[0]["content"], repr(out[0]["content"]))
check("rest of text kept", out[0]["content"].endswith("Say OK."))

print("2. the reported degeneration: 500 BOS in one assistant turn")
m = [{"role": "user", "content": "hi"},
     {"role": "assistant", "content": BOS * 500},
     {"role": "user", "content": "Say OK."}]
out = strip(m)
check("all 500 removed", BOS not in out[1]["content"], f"len={len(out[1]['content'])}")
enc = encoding_dsv4.encode_messages(out, thinking_mode="chat",
                                    add_default_bos_token=False)
check("encoder output has no BOS", BOS not in enc, f"count={enc.count(BOS)}")

print("3. BOS in several messages (the propagating loop)")
m = [{"role": "user", "content": BOS + "a"},
     {"role": "assistant", "content": BOS * 3},
     {"role": "user", "content": BOS + "b"}]
out = strip(m)
enc = encoding_dsv4.encode_messages(out, thinking_mode="chat",
                                    add_default_bos_token=False)
check("no BOS survives encoding", BOS not in enc, f"count={enc.count(BOS)}")

print("4. must NOT change: no BOS present")
m = [{"role": "system", "content": "You are helpful."},
     {"role": "user", "content": "hello"}]
out = strip(m)
check("identity when clean", out is m, "returns the same object")

print("5. must NOT change: roles, ids, tool fields")
m = [{"role": "assistant", "content": BOS + "answer", "tool_call_id": "call_1",
      "name": "read_file"}]
out = strip(m)
check("role preserved", out[0]["role"] == "assistant")
check("tool_call_id preserved", out[0].get("tool_call_id") == "call_1")
check("name preserved", out[0].get("name") == "read_file")

print("6. must NOT mutate the caller's list")
m = [{"role": "assistant", "content": BOS * 3}]
_ = strip(m)
check("input list untouched", m[0]["content"].count(BOS) == 3,
      f"still has {m[0]['content'].count(BOS)}")

print("7. non-string content is left alone")
m = [{"role": "user", "content": [{"type": "text", "text": BOS + "x"}]}]
out = strip(m)
check("list content untouched", out[0]["content"][0]["text"].count(BOS) == 1)

print("8. the fix is what the encoder alone does not do")
m = [{"role": "user", "content": BOS + "Say OK."}]
enc_raw = encoding_dsv4.encode_messages(m, thinking_mode="chat",
                                       add_default_bos_token=False)
check("encoder alone STILL leaks BOS", BOS in enc_raw,
      "-> confirms the sanitiser is doing real work")

print()
print(f"FAILS = {fails}")
sys.exit(1 if fails else 0)
