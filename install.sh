#!/bin/bash
set -o -x
cd /data/nvme/sglang-codex
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PWD/.venv/bin:$PATH
export UV_LINK_MODE=copy
export SGLANG_BUILD_RUST_EXTS=none
~/.local/bin/uv pip install --python .venv/bin/python --prerelease=allow -e "sglang/python"
echo "INSTALL_RC=$?"
