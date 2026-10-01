#!/bin/bash
cd /data/nvme/sglang-codex
export MAXREQ=2 GRAPH_MAX_BS=2 GRAPH_BS="1 2"
bash ./serve_dummy.sh 2 4 smoke-dummy2
