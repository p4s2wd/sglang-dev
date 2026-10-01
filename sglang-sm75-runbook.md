# DeepSeek-V4-Flash on 8× RTX 2080 Ti — 运行手册（Runbook）

> 配套：`sglang-sm75-plan.md`（计划）、`sglang-sm75-progress.md`（进度）。
> 代码：`sglang` 仓库分支 `sm75-dsv4-flash`（基于上游 `be00a543a7`，2 个 commit：`a85da5470b` 内核 + `7abfe29cb7` 接线）。
> 权重：`/data/nvme/models/DeepSeek/V4/DeepSeek-V4-Flash-0731`（HF 格式，155.4 GiB）。
> 最近更新：2026-09-14（S3）

---

## 0. 一页纸速查

| 项 | 值 |
|---|---|
| 启动开关 | `SGLANG_ALLOW_SUB80_QUANT=1`（唯一必须手动设的；其余 sub-90 env 由 model_hook 自动设并打 WARNING 日志） |
| 首选并行 | `--tp 4 --pp 2`（朋友方案）；对比 `--tp 2 --pp 4`（权重驻留减半） |
| KV 池 | `--kv-cache-dtype fp8_e4m3`（V4 paged KV 原生 FP8 布局，store 内核已验证 sm_75 可跑） |
| 运行 dtype | fp16（SM75 无原生 BF16，loader 自动降级，无需干预） |
| MoE 后端 | triton（自动）；**W4A16 默认走手写 PTX mma.sync 内核（53 GB/s，6.6× Triton）**，JIT 在权重加载期编译预热（需 nvcc+ninja），失败自动回退 Triton（8–12 GB/s） |
| 预期 decode | 粗估 30–60 tok/s（TP4+PP2，PTX 内核，未实测；B=1 单卡 MoE 层 ~1.4ms × 43 层 + 注意力） |
| 冒烟顺序 | ① import 冒烟 → ② 内核单测 → ③ 2 卡小配置起服务 → ④ 8 卡全量 |

## 1. 环境要求（目标机）

- 驱动支持 CUDA ≥ 12.6（JIT 内核用 nvcc 编译；分析机用 `/usr/local/cuda-12.6`，目标机 `nvcc --version` 确认）
- `ninja` 在 PATH（JIT 编译依赖；conda 环境 `pip install ninja` 即可）
- torch ≥ 2.7 + triton ≥ 3.3（分析机 torch 2.13/triton 3.7 验证通过）
- `sgl_kernel` 必须可装/可导入（目标机是完整安装，不像分析机绕开它）
- 磁盘：加载检查点需 ~160G 顺序读 + 页缓存；NVMe 直连

## 2. 冒烟步骤（严格按序，每步都是上一步的前提）

### 2.1 import 冒烟（消掉"接线从未执行"风险）
```bash
cd sglang && pip install -e python --no-deps 2>/dev/null || true   # 或按目标机安装方式
python -c "import sglang.srt; print('import OK')"
```
**预期**：无 ImportError。若报 `deep_gemm` 缺失——正常，sub-90 路径全部惰性导入且已加开关；报 `tilelang` 同理。

### 2.2 内核单测（目标机 GPU 上重跑分析机验证）
```bash
CUDA_VISIBLE_DEVICES=0 python sglang-audit/test_mxfp4_w4a16.py
CUDA_VISIBLE_DEVICES=0 python sglang-audit/test_moe_sub80_e2e.py
CUDA_VISIBLE_DEVICES=0 python sglang-audit/test_mqa_logits.py
CUDA_VISIBLE_DEVICES=0 python sglang-audit/test_mqa_logits_real_shape.py
CUDA_VISIBLE_DEVICES=0 python sglang-audit/test_real_weights.py      # 需检查点路径
PATH=$CONDA_PREFIX/bin:$PATH CUDA_VISIBLE_DEVICES=0 python sglang-audit/test_jit_store_real.py
PATH=$CONDA_PREFIX/bin:$PATH CUDA_VISIBLE_DEVICES=0 python sglang-audit/test_jit_qindexer_real.py
# B8 PTX W4A16 内核（需 nvcc+ninja 在 PATH）：
PATH=$CONDA_PREFIX/bin:$PATH CUDA_VISIBLE_DEVICES=0 python sglang-audit/b8/test_w4a16_ptx.py
PATH=$CONDA_PREFIX/bin:$PATH CUDA_VISIBLE_DEVICES=0 python sglang-audit/b8/test_direct.py
PATH=$CONDA_PREFIX/bin:$PATH CUDA_VISIBLE_DEVICES=0 python sglang-audit/b8/test_e2e_ptx.py
```
**预期**：8 项 PASS（分析机 2080 Ti 全过；目标机同为 SM75 应一致）。PTX 三项若报 nvcc/ninja 缺失，说明 JIT 工具链没配好——服务仍会回退 Triton 但慢 6.6×。

### 2.3 拓扑确认（B7 显存编排的输入）
```bash
nvidia-smi topo -m                      # NVLink 对（预期每 2 卡一对）
nvidia-smi --query-gpu --query-gpu-name,driver_version,memory.total
```
把输出回传到进度文件 §3。

### 2.4 起服务（先小后大）
```bash
# 第一步：2 卡最小配置（1 个 PP stage 的注意力 + 部分专家），验证加载链路
SGLANG_ALLOW_SUB80_QUANT=1 python -m sglang.launch_server \
  --model /data/nvme/models/DeepSeek/V4/DeepSeek-V4-Flash-0731 \
  --tp 2 --pp 1 --mem-fraction-static 0.90 --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-max-bs 1 --max-running-requests 1 --context-length 8192
```
**注意**：`--pp 1` 时权重 155.4G 放不进 2×22G——此步仅当**加载器/量化门/内核 dispatch 冒烟**用，预期在显存分配处 OOM 即算通过（看日志走到 `KV cache` 分配且无 quant/dispatch 报错）。真正跑通从下面 8 卡开始。

```bash
# 正式：TP4+PP2（朋友方案）
SGLANG_ALLOW_SUB80_QUANT=1 python -m sglang.launch_server \
  --model /data/nvme/models/DeepSeek/V4/DeepSeek-V4-Flash-0731 \
  --tp 4 --pp 2 --mem-fraction-static 0.90 --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-max-bs 4 --max-running-requests 4 --context-length 32768
```

### 2.5 出字验证 + 探针
```bash
curl http://127.0.0.1:30000/generate -d '{"text":"介绍一下你自己","sampling_params":{"temperature":0}}'
# 性能画像（B7 输入）：
python sglang-audit/probe/probe_prefill_decode.py --base-url http://127.0.0.1:30000 --out ./probe-out-tp4pp2
# 换 --tp 2 --pp 4 再跑一遍 → probe-out-tp2pp4，对比报告进进度文件
```

## 3. 显存预算（每卡 22.0 GiB，2080 Ti）

| 项 | TP4+PP2 | TP2+PP4 |
|---|---|---|
| 权重（MXFP4 packed + FP8 注意力 + fp16 反量化增量） | ~19.4 GiB | ~19.4 GiB |
| KV（FP8 paged，32k ctx × 4 req 量级） | ~1.2 GiB | ~1.2 GiB |
| 激活/工作区/CUDA graph | ~1.0 GiB | ~1.0 GiB |
| 余量 | ~0.4 GiB ⚠️ 紧 | ~0.4 GiB ⚠️ 紧 |

- 权重是瓶颈（155.4/8 = 19.4 GiB/卡，与 TP/PP 切法无关——PP 切层、TP 切宽，总量不变）。
- 若 OOM：先降 `--context-length`/`--max-running-requests`，再试 `--mem-fraction-static 0.92`；最后手段 `--disable-cuda-graph`（省 graph 池 ~0.5 GiB，decode 变慢）。
- FP8 注意力反量化到 fp16 的增量（+1.8 GiB/卡）已含在上表；若走稠密 FP8 线性层反量化路径，再 +~1 GiB/卡——TP2+PP4 下更紧。

## 4. 故障排查（按日志特征定位）

| 日志/现象 | 根因 | 处置 |
|---|---|---|
| `quantization requires capability >= 8.0` | 没设 `SGLANG_ALLOW_SUB80_QUANT=1` | 加上 |
| 无 `sub-90 path` WARNING | model_hook 未命中（非 DeepseekV4 架构或 major≥9） | 确认 `--model` config 的 architectures |
| `deep_gemm` ImportError | 正常（sub-90 不该导入）；若真导入说明某开关没生效 | 查 `SGLANG_OPT_DSV4_NONPAGED_INDEXER` 等是否被手动覆盖 |
| Triton `out of resource: shared memory` | BLOCK 超 64KB | 用默认 BM16/BN64/BK128/nw8；MQA logits 的 BLOCK_H=8 别改大 |
| `tl.float8e4nv` 编译错 | 某路径漏了 torch 预反量化 | 记录栈，按 B4 模式改（torch `.to(fp16)` 预转） |
| MoE 输出 NaN | W4A16 k 越界读（已修）或 scale dtype 不符 | 确认 scale 是 float32 真值（loader copy_ 自动转） |
| JIT `ninja not found` | PATH 缺 ninja | `pip install ninja` 或加 conda bin |
| decode 极慢（<1 tok/s） | 正常量级——W4A16 12 GB/s 天花板 | 要质变走 B8（inline-PTX）或 Track A（llama.cpp mmvq） |
| CUDA graph 捕获失败 | 某内核不可捕获（torch 兜底路径有动态 shape） | `--disable-cuda-graph` 先跑通，再定位具体内核 |
| 日志 `PTX W4A16 kernel unavailable` | JIT 编译失败（nvcc/ninja 不在 PATH，或 CUDA 版本不匹配） | 配好工具链；否则回退 Triton（慢 6.6×，功能不受影响） |
| MoE 比预期慢 6.6× | PTX 内核未启用（JIT 失败或 TP 切分后 N%8/K%32 不满足） | 查上面一条；TP8 时 inter=2048/8=256（%8=0 ✓）、K/2=2048/8=256（%16=0 ✓）一般满足 |

## 5. 已知限制（诚实清单）

1. **性能**：W4A16 12 GB/s（Triton 在 Turing 上 dequant→dot 必经 SMEM 往返的硬限制）。B=1 单卡 MoE 层 9.4ms。8 卡 decode 粗估 10–20 tok/s，**未实测**。
2. **端到端未验证**：接线代码只做过 py_compile + 内核级测试；2.1–2.5 就是为此设计，首次上 8 卡机严格按序。
3. **CUDA graph**：torch 兜底路径（paged logits / sparse MLA）的动态 shape 可能捕获失败——先 `--disable-cuda-graph` 跑通。
4. **数值**：fp16 运行（非 bf16），HC/sinkhorn 对拍（B6）未做；长上下文稳定性未验证。
5. **Track A**（llama.cpp）：A1 GGUF 已就绪（`gguf-out/DSV4-Flash-0731-mxfp4_moe.gguf`），A2–A4 同样需目标机。

## 6. 回退

- 全部改动 gated：不设 `SGLANG_ALLOW_SUB80_QUANT` 时行为与上游一致。
- 回退单个 commit：`git revert 7abfe29cb7 a85da5470b`（先接线后内核）。
- 单开关回退：model_hook 的 sub-90 分支在 `arg_groups/model_hook.py` DeepseekV4 段 `else: major<9`；内核 dispatch 在 `fp8.py` `_apply_sub80_mxfp4_w4a16` 与 `dsv4/indexer.py` `_forward_nonpaged_indexer`。
