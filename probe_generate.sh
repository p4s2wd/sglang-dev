#!/bin/bash
# Send one generate request and print the raw response.
# Usage: ./probe_generate.sh [prompt] [max_new_tokens]
PROMPT="${1:-The capital of France is}"
MAXTOK="${2:-16}"
PORT="${PORT:-30000}"
PY=/data/nvme/sglang-codex/.venv/bin/python
BODY=$("$PY" - "$PROMPT" "$MAXTOK" <<'PY'
import json, sys
print(json.dumps({
    "text": sys.argv[1],
    "sampling_params": {"temperature": 0.0, "max_new_tokens": int(sys.argv[2])},
}))
PY
)
curl -s -m 120 "http://127.0.0.1:${PORT}/generate" \
  -H "Content-Type: application/json" \
  -d "$BODY"
echo
