#!/bin/bash
# Show the interesting lines of a serve log, skipping the giant server_args dump.
LOG="${1:-/data/nvme/sglang-codex/logs/serve-tp2pp4.log}"
grep -av "server_args=" "$LOG" \
  | grep -aE "Load weight|Finished|KV cache|kv cache|max_total|avail mem|Scheduler hit|OutOfMemory|OutOfResources|Traceback|Error|error|WARNING|Capture|capture|ready|Uvicorn|POST /generate|Generate|throughput|tok/s|shards: 100%" \
  | tail -n "${2:-40}"
