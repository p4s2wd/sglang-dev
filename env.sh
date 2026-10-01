# SM75 / DeepSeek-V4-Flash environment for myai001 (8x RTX 2080 Ti)
export SGLANG_REPO=/data/nvme/sglang-codex/sglang
export DSV4_CKPT=/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-0731
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:/data/nvme/sglang-codex/.venv/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export SGLANG_ALLOW_SUB80_QUANT=1
export TORCH_CUDA_ARCHS=7.5
export PYTHONUNBUFFERED=1
source /data/nvme/sglang-codex/.venv/bin/activate
