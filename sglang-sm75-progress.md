# SGLang SM75 × DeepSeek-V4-Flash 适配进度表

> 进度文件（每会话更新）。计划文件：`sglang-sm75-plan.md`（任务 ID 与其一致）。
> 最近更新：2026-09-14（会话 S2）

---

## 0. 状态快照（会话开始时先读这里）

- **当前阶段**：M0 部分完成——S2/S3 已在分析机（2×4090D + 1×2080Ti）完成 A1、B1–B5、C1 的代码、单测与真实权重验证；等待目标 8 卡机做端到端验收（M1–M6）
- **当前阻塞**：
  1. ⛔️ ~~B0 等朋友 fork~~ **已关闭（S3）**：用户确认无法获取朋友 fork/diff——`sm75-dsv4-flash` 分支即唯一实现，B0 不再阻塞任何事
  2. ⛔️ 目标 8 卡机的拓扑确认（需在该机执行 `nvidia-smi topo -m`；当前分析机只有 2×4090D + 1×2080Ti，且 NVML 初始化失败）
  3. ⛔️ A2/A3/A4、B7、B8、M1–M6 全部需要目标 8 卡机
- **S2 已完成（本机可验证部分全部完成）**：A1 GGUF 转换（156.4G 产物已验证）；B1 量化门；B2 MXFP4 W4A16 Triton 内核+接线+e2e 测试 PASS；B3 FP8 稠密/wo_a 加载期反量化+bf16 硬编码修复；B4 Triton MQA logits 内核（PASS）+ 非分页接线 + 复用 sm120 torch 稀疏 MLA；B5 model_hook sub-90 分支；C1 profiler 探针（合成 trace 验证通过）；另验证 JIT CUDA 内核（store/indexer-Q）在 sm_75 可编译
- **S3 已完成（2080 Ti 局部验证）**：① **真实权重对拍 PASS**——从检查点 mmap 读 layer-2 真实 MXFP4 专家（w1=gate/w3=up/w2=down 三件套，加载时融合 w13），dequant 落在 e2m1 网格、单专家 GEMM rel 3.7e-4、6 专家 topk=3 MoE 完整回放 rel 5.5e-4；② **JIT 内核实跑 PASS**——store（FP8 量化误差 3.7%=e4m3 正常、rope bf16 直通、UE8M0 scale 正确）+ indexer-Q（RoPE+Hadamard+fp8，真实 H=64/D=128）；③ **MQA logits 真实形状 PASS**——H=64 头超 SMEM，重构为头维分块（BLOCK_H=8）后 rel 1.6e-7；④ **W4A16 性能调优**——发现 Triton 在 Turing 上「寄存器算出的 b 喂 tl.dot 必经共享内存往返」硬限制（Marlin 靠 inline PTX 绕开，属 B8 wheel 活），Triton 天花板 ~12 GB/s；通过 coalescing（k 维放最后维）+ `num_warps=8`（dequant 是 ALU 密集，4 warp 藏不住延迟）把内核从 3→12 GB/s（4×），B=1 MoE 层 9.4ms
- **关键事实速查**：权重 155.4 GiB（FP8 注意力 14.5 GiB + MXFP4 专家 138.0 GiB）；2080Ti=SM75 无 FP8 TC/无原生 BF16（torch 自动降级 fp16）；NVLink 2 卡/对；SM75 Triton 无 e4m3 位转（`tl.float8e4nv` 编译期报错，需 torch 预反量化）；SM75 共享内存 64KB（Triton 内核 BLOCK 需相应缩小）；**2080 Ti 实测带宽 543 GB/s（DtoD）/ fp16 GEMM 48 TFLOP/s**；**W4A16 Triton 内核最优配置 BM=16/BN=64/BK=128/nw=8/ns=2**；sglang 分支 `sm75-dsv4-flash`（基于 `be00a543a7`）
- **运行方法（目标机）**：`SGLANG_ALLOW_SUB80_QUANT=1 python -m sglang.launch_server --model <HF路径> --tp 4 --pp 2 --kv-cache-dtype fp8_e4m3 ...`（sub-90 env 组合由 model_hook 自动设置）

## 1. 任务状态表

### Track A — llama.cpp GGUF 路径

| ID | 任务 | 状态 | 完成度 | 证据/产出 | 备注 |
|---|---|---|---|---|---|
| A1 | HF→GGUF 转换（mxfp4 专家保真） | **done** | 100% | `/home/shuang/codex/gguf-out/DSV4-Flash-0731-mxfp4_moe.gguf`（156.4G，GGUF v3，1328 tensors，gguf-py 读回验证 architecture=deepseek4） | 命令：`convert_hf_to_gguf.py <HF> --outtype f16 --use-temp-file`；日志 `gguf-out/convert.log` |
| A2 | 8 卡加载（split-mode layer/tensor × tensor-split） | pending | 0% | — | 需目标机 |
| A3 | 正确性对拍（logprob 漂移 + perplexity） | pending | 0% | — | 需参考输出（H200 或 llama.cpp CPU 小样本） |
| A4 | llama-bench 基线 | pending | 0% | — | 需目标机 |

### Track B — SGLang SM75 适配

| ID | 任务 | 状态 | 完成度 | 证据/产出 | 备注 |
|---|---|---|---|---|---|
| B0 | 获取朋友 fork + rebase `be00a543a7` | **closed** | — | 用户确认无法获取 fork/diff（S3）；`sm75-dsv4-flash` 分支（2 commit：`a85da5470b` 内核 + `7abfe29cb7` 接线）即唯一实现 | 不再阻塞任何任务 |
| B1 | 量化门参数化（`SGLANG_ALLOW_SUB80_QUANT`） | **done** | 100% | `environ.py`（新增 EnvBool）+ `model_loader/loader.py`（capability 门改可旁路，带告警） | 默认 False 保持上游行为 |
| B2 | MXFP4→FP16 Triton W4A16 分组 GEMM + MoE 接线 | **done** | 100% | 内核 `kernels/ops/moe/mxfp4_w4a16_kernels.py`（e2m1 位构造寄存器反量化 + coalesced [N,KH] 加载 + interleave，fp32 累加，最优 BM16/BN64/BK128/nw8）；接线 `fp8.py` `Fp8MoEMethod._apply_sub80_mxfp4_w4a16`；测试 `test_mxfp4_w4a16.py`（f32/u8 双 PASS）+ `test_moe_sub80_e2e.py`（rel 8e-4）+ **`test_real_weights.py`（检查点真实专家权重对拍 rel 5.5e-4）** + `bench_w4a16.py`（12 GB/s，B=1 MoE 层 9.4ms） | 权重保持 packed（138 GiB 不反量化）；scale float32 真值（loader copy_ 自动转）；修 BLOCK_K>K 越界读；**Triton 在 Turing 上 dequant→dot 必经 SMEM 往返=12 GB/s 天花板，Marlin inline-PTX 可绕开（B8 wheel 活）** |
| B3 | FP8 注意力 W8A16 兜底（wo_a + 稠密反量化） | **done** | 100% | `deepseek_v4.py`：`_dequant_fp8` 改 `get_default_dtype()`（SM75→fp16）、wo_a `params_dtype` 同步；`fp8.py` `Fp8LinearMethod`：sub80 加载期反量化（`_dequantize_layer_to_16bit`，支持 128×128 block scale）+ `apply` 直通 `F.linear`；model_hook 自动 `SGLANG_OPT_FP8_WO_A_GEMM=0` | FP8 部分仅 14.5 GiB，反量化后每卡 +1.8 GiB 可接受；稠密 FP8 scale 是 float32（copy_ 自动转换） |
| B4 | DSA Triton FP16（MQA logits + 稀疏 MLA decode） | **done** | 100% | 内核 `kernels/ops/attention/dsa/triton_mqa_logits.py`（torch 预反量化 fp16 + Turing TC dot，**头维分块 BLOCK_H=8 寄存器累加**——真实 H=64 头 q tile 128KB 超 SM75 64KB SMEM）；接线 `dsv4/indexer.py` `_forward_nonpaged_indexer`（无 DeepGEMM 时走 Triton）；稀疏 MLA decode 复用 `flash_mla_sm120.py` torch 路径（`deepseek_v4_backend.py` `_use_torch_sparse_mla()` 把 sub-90 并入 sm120 分支）；测试 `test_mqa_logits.py`（rel 1e-7）+ **`test_mqa_logits_real_shape.py`（真实 H=64/D=128/K=4096 rel 1.6e-7）** + **JIT store/indexer-Q 真实形状实跑 PASS** | prefill 走非分页 Triton 内核；`test_jit_store_real.py` 验证 FP8 量化误差 3.7%（e4m3 正常）+ rope bf16 直通 + UE8M0 scale 正确；`test_jit_qindexer_real.py` 验证 RoPE+Hadamard+fp8 输出有限 |
| B5 | 后端选择链加 sub-90 分支 | **done** | 100% | `arg_groups/model_hook.py` DeepseekV4 分支新增 `else: major<9`：关 FP8 wo_a/DeepGEMM HC/TileLang 全家/topk_v2/JIT indexer metadata/多流重叠，开 `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`，moe_runner→triton，打告警日志 | dsv4 注意力后端无 arch 断言（fa3 断言不在 V4 路径上）；CuTeDSL/AOT 路径经 `paged_mqa_logits_backend.py` 枚举天然不可达 |
| B6 | FP16 数值对拍（HC/sinkhorn/sqrtsoftplus） | pending | 0% | — | 依赖目标机（或本机 2080Ti 跑小模型对拍，待做） |
| B7 | 显存编排（TP4+PP2 vs TP2+PP4 对比） | pending | 0% | — | 依赖 C1 + 目标机（B0 已关闭，不再是前置） |
| B8 | PTX W4A16 内核（性能部分）+ sm75 wheel 重编 | **部分 done** | 60% | **内核完成**：`jit/csrc/moe/mxfp4_w4a16_ptx{,_direct}.cuh`（手写 `mma.sync.aligned.m16n8k8`，寄存器内 e2m1 反量化，零共享内存）；direct 版直读原始 packed 权重（**无 repack buffer**，避免 22G 卡 OOM）53 GB/s=**6.6× Triton**；repacked 版 75 GB/s=9×；接线 `fp8.py` dispatch（PTX 优先 + TP 对齐校验 N%8/K%32 + Triton 回退）+ 加载期 JIT 预热（避开 CUDA graph capture）；测试 `sglang-audit/b8/{test_w4a16_ptx,test_direct,test_e2e_ptx}.py` 全 PASS（rel 3.4e-4/3.6e-4/6.1e-4）；commit `524aab411d` | **wheel 重编 + CI 仍需目标机**（`TORCH_CUDA_ARCHS=7.5`）；关键发现：Turing 是 k=8（`m16n8k16` 需 sm_80+）；scale 必须烘进 B fragment（C 的列是 2c4 不是 gid）；repack 双缓冲在 8 卡机必 OOM（17.25+17.25>22 GiB）故默认 direct 版 |

### Track C — Profiler 探针

| ID | 任务 | 状态 | 完成度 | 证据/产出 | 备注 |
|---|---|---|---|---|---|
| C1 | `sglang-audit/probe/probe_prefill_decode.py` | **done** | 100% | 复用 `/start_profile`（profile_by_stage + detailed_annotations + num_steps 自动停）→ 轮询 `*.trace.json.gz` → 按算子类别（GEMM/MoE/Attention/Comm/Memcpy/Norm）分解 prefill/decode GPU 时间 + top-12 内核；合成 trace 解析验证通过 | 用法：`python probe_prefill_decode.py --base-url http://127.0.0.1:30000 --out ./probe-out`；`--traces-only` 可离线解析 |
| C2 | 阈值规则 + 报告模板 | pending | 0% | — | 依赖 C1 + 目标机数据 |

### 里程碑

| ID | 内容 | 状态 | 完成度 |
|---|---|---|---|
| M0 | fork + 拓扑确认 | **部分**（fork 已关闭=本分支唯一实现；剩拓扑确认待目标机） | 50% |
| M1 | Track A 出字 | pending | 0%（A1 产物就绪） |
| M2 | Track A 验收 | pending | 0% |
| M3 | B1+B2+B3 合流 | pending | 60%（代码+单测完成，待目标机端到端） |
| M4 | B4+B5 合流 + 对比报告 | pending | 50%（代码+单测完成，待目标机） |
| M5 | B6+B7 稳定服务 | pending | 0% |
| M6 | B8+C2 回归化 | **部分** | 30%（B8 内核完成并接线；wheel 重编 + CI + C2 待目标机） |

## 2. 会话日志

| 日期 | 会话 | 改动/结论 | 证据 |
|---|---|---|---|
| 2026-09-13 | S1 | 立项：确认 B1–B8 全部阻塞点（含权重 dtype 实测：I8=138.0 GiB、F8_E4M3=5.9 GiB、F8_E8M0=8.6 GiB，total_size=166,878,536,440）；确认 llama.cpp 侧 DSV4 全链路存在（deepseek4.cpp/kv-cache-dsv4.cpp/conversion/deepseek.py:528）且 Turing 内核一等公民（GGML_CUDA_CC_TURING=750，mmvq/fattn-mma-f16 有专用分支）；确认 SGLang 已有 profiler stage 级基建（io_struct.py:2117）；评估朋友 TP4+PP2+NVLink KV 对半方案（正确但非充分，瓶颈是权重 19.4 GiB/卡；提出 TP2+PP4 变体待对比）；产出计划文件+本进度文件 | 计划 §1 全表 file:line |
| 2026-09-14 | S2 | **A1 完成**：GGUF 156.4G 转换+验证。**B1–B5+C1 代码完成**（sglang 分支 `sm75-dsv4-flash`，7 改 2 新）：B1 量化门旁路；B2 MXFP4 W4A16 Triton 内核（寄存器 e2m1 反量化）+ `Fp8MoEMethod` sub80 分支接线，e2e 测试 PASS（rel 8e-4）；B3 稠密 FP8 加载期反量化到 fp16 + wo_a bf16 硬编码改 default dtype；B4 Triton MQA logits 内核（PASS rel 1e-7）+ 非分页接线 + `_use_torch_sparse_mla()` 复用 sm120 torch 稀疏 MLA；B5 model_hook sub-90 env 组合分支；C1 探针脚本（合成 trace 验证）。**关键发现**：SM75 Triton 无 e4m3 位转（需 torch 预反量化）；SM75 共享内存 64KB；JIT CUDA 内核（store/indexer-Q，含 cuda_fp8.h 软件模拟）sm_75 可编译；fp4 scale 经 loader copy_ 自动变 float32 真值；DeepGEMM 在 SM<90 自动禁用；`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` 已提供 decode 分页兜底；共享专家 FP8 与路由 MXFP4 混格——sub-80 必须保持共享专家融合关闭（默认即关） | `sglang-audit/test_*.py` 全 PASS；`git status` 9 文件 |
| 2026-09-14 | S3 | **2080 Ti（GPU 2）局部验证**：① 真实权重对拍 `test_real_weights.py` PASS——mmap 读检查点 layer-2 真实 MXFP4 专家（确认检查点布局 w1=gate/w3=up/w2=down，sglang 融合 w13），dequant 落 e2m1 网格、单专家 GEMM rel 3.7e-4、6 专家 topk=3 MoE 完整回放 rel 5.5e-4；② JIT 内核实跑 PASS——`test_jit_store_real.py`（真实 SWA 页布局 stride 74880，FP8 量化误差 3.7%=e4m3 正常、rope bf16 直通、UE8M0 scale 正确）、`test_jit_qindexer_real.py`（真实 H=64/D=128/rope64，RoPE+Hadamard+fp8 输出有限）；③ `test_mqa_logits_real_shape.py`——H=64 头 q tile 128KB 超 SMEM，重构内核为头维分块（BLOCK_H=8 寄存器累加）后 PASS rel 1.6e-7；④ **W4A16 性能调优** `bench_w4a16.py`+消融——定位 Triton 在 Turing 上「寄存器算出的 b 喂 tl.dot 必经 SMEM 往返」硬限制（纯 fp16 GEMM 16 GB/s vs dequant 4 GB/s），Triton 天花板 ~12 GB/s；落地 coalescing + `num_warps=8` + 位构造 decode，内核 3→12 GB/s（4×）；⑤ 带宽基线 `bw_probe.py`：DtoD 543 GB/s、fp16 GEMM 48 TFLOP/s；**收尾**：B0 关闭（无 fork，本分支即唯一实现）；commit `a85da5470b`+`7abfe29cb7`；`flash_mla_sm120.py` 返回 dtype 改 q.dtype；产出 `sglang-sm75-runbook.md` | `sglang-audit/{test_real_weights,test_jit_store_real,test_jit_qindexer_real,test_mqa_logits_real_shape,bench_w4a16,bw_probe,ablate*}.py` |
| 2026-09-14 | S4 | **B8 性能内核完成（PTX 手写 mma.sync）**：绕开 Triton 在 Turing 上的 SMEM 往返硬限制。`jit/csrc/moe/mxfp4_w4a16_ptx.cuh`（repacked 布局，75 GB/s=9×）+ `mxfp4_w4a16_ptx_direct.cuh`（**直读原始 packed 权重，无 repack buffer**，53 GB/s=6.6×）——direct 为默认，因 repack 双缓冲在 8 卡机必 OOM（每卡 17.25 GiB packed + 17.25 GiB repacked > 22 GiB）。接线 `fp8.py`：dispatch 优先 PTX-direct（TP 对齐校验 N%8/K%32，不满足回退 Triton）+ `process_weights_after_loading` 预热 JIT（避免 nvcc 在 CUDA graph capture 内编译）。**调试中修的 3 个真 bug**：Turing 是 `m16n8k8`（k=8，`m16n8k16` 需 sm_80+）；scale 必须烘进 B fragment（C fragment 的列是 2c4 而非 gid，乘累加器会错）；删 scale 时误删累加行。测试 `sglang-audit/b8/{test_w4a16_ptx,test_direct,test_e2e_ptx}.py` 全 PASS（rel 3.4e-4/3.6e-4/6.1e-4）；全套 8 项回归 PASS。commit `524aab411d` | `sglang-audit/b8/` 3 测试 + `git log` |

## 3. 待办速查（下次会话直接认领）

1. ~~[等用户] 提供朋友 fork~~ **已关闭（S3）**：无 fork，本分支即唯一实现
2. [等用户/目标机] 执行 `nvidia-smi topo -m` 与 `nvidia-smi --query-gpu --query-gpu-name,driver_version,memory.total` 并回传 → 输入 B7
3. [目标机·第一优先] 按 `sglang-sm75-runbook.md` §2 冒烟序：import 冒烟 → 7 项内核单测 → 起服务 → 出字 → C1 探针（消掉"接线从未执行"风险）
4. [目标机] M1：llama.cpp 加载 A1 GGUF 出字；M3/M4：TP4+PP2 vs TP2+PP4 对比报告
5. [目标机] B6 数值对拍（本机不可做：缺 sgl_kernel，无法构造真实 layer）
6. ~~[可选·本机] B8 预研~~ **已完成（S4）**：PTX-direct W4A16 内核 6.6×，已接线并 commit `524aab411d`；剩 wheel 重编需目标机

## 4. 交付物清单（S2–S4）

- `sglang` 分支 `sm75-dsv4-flash`：`a85da5470b`（2 个 Triton 内核）+ `7abfe29cb7`（8 文件接线）+ `524aab411d`（PTX W4A16 内核 + dispatch）
- `gguf-out/DSV4-Flash-0731-mxfp4_moe.gguf`（156.4 GiB，A1）
- `sglang-sm75-runbook.md`（目标机手册：冒烟序/显存预算/故障排查/回退）
- `sglang-audit/`：8 个 PASS 测试（含 `b8/` 3 个）+ bench/ablate/probe 工具集

