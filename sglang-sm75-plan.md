# SGLang SM75 (8×RTX 2080Ti 22G) × DeepSeek-V4-Flash 适配计划

> 计划文件（稳定参考，改动需走"计划修订"记录）
> 进度文件：`sglang-sm75-progress.md`（每会话更新，反映当前适配情况）
> 创建：2026-09-13 ｜ 基线：sglang `be00a543a7`、llama.cpp `73a43d1f6`
> 模型：`/data/nvme/models/DeepSeek/V4/DeepSeek-V4-Flash-0731`（155.4 GiB，FP8 注意力 + MXFP4 打包专家）
> 硬件：8×RTX 2080Ti 22G（SM 7.5 / Turing，无 FP8 TC、无原生 BF16），NVLink 桥按 **2 卡/对** 互联（4 对）

---

## 1. 阻塞点清单（B1–B8，均有代码证据）

| # | 阻塞点 | 证据（file:line） | 严重度 |
|---|---|---|---|
| B1 | **FP8 量化加载门**：`Fp8Config.get_min_capability()` 返回 80，加载前比较算力，75 < 80 直接抛 `ValueError` | `srt/layers/quantization/fp8.py:299-311`；`srt/model_loader/loader.py:243-254` | 硬拦截 |
| B2 | **MXFP4 专家无 SM75 内核**：专家权重实测为 I8 打包 MXFP4（138.0 GiB / 72,317 张量中 35,328 个 I8）。Marlin 对 capability<80 返回空支持表；FlashInfer MXFP4/CUTLASS/TRT-LLM 仅 SM90/100/120；DeepGEMM FP4 仅 SM90+ | `srt/layers/quantization/marlin_utils.py:77-78`；cookbook `DeepSeek-V4.mdx:374`（Hopper 用 Marlin W4A16）；权重实测 | 硬拦截 |
| B3 | **Turing 无 FP8 Tensor Core**：V4 注意力 wo_a 走 DeepGEMM `fp8_einsum`，代码只有 sm100（ue8m0）/sm90（fp32 scale）两个分支；fused RMSNorm+FP8 量化同样只有 SM90+/ROCm 路径 | `srt/models/deepseek_v4.py:1831-1847`、`:1307`、`:1424` | 硬拦截 |
| B4 | **DSA 稀疏注意力内核缺失**：paged-MQA logits 后端只有 `deepgemm`/`cutedsl`(SM100)/`aiter`(ROCm) 三选，无 Triton 兜底；MLA 内核为 `sparse_mla_q8kv8_prefill_sm90`、`flash_mla_sm120` | `srt/layers/attention/dsa/paged_mqa_logits_backend.py:26-45`；`dsa_backend.py:528,2740,2935` | 硬拦截 |
| B5 | **注意力后端选择链无 SM75 分支**：fa3 断言 `major==9`；flashmla/cutlass_mla 为 SM90+；DSA 未接任何通用后端 | `srt/layers/attention/attention_registry.py:185-189,208-215` | 硬拦截 |
| B6 | **Turing 无原生 BF16**：SGLang 对 major<8 降级 FP16（`maybe_downgrade_dtype_for_legacy_gpu`），但 V4 内核（HC/sinkhorn、fp8 quant 融合）按 BF16/FP8 写，FP16 下溢出与精度风险未验证 | `srt/model_executor/model_runner_components/load_model_utils.py:75-86` | 软风险 |
| B7 | **显存预算超限**：权重 155.4 GiB ÷ 8 = **19.4 GiB/卡**，剩 2.6 GiB 要装 KV + 激活 + CUDA graph + NCCL 缓冲。V4 压缩 KV 本身很小（compress_ratios 4/128，128K ctx 约 0.5–1 GiB），但 graph 捕获 + NCCL 就要 1.5–2.5 GiB | 权重 index 实测 `total_size=166,878,536,440`；`config.json` compress_ratios | 硬约束 |
| B8 | **内核二进制无 SM75**：发行 wheel/Docker 镜像按 SM90/100/120 编译（cookbook 支持硬件清单：h100/h200/b200/b300/gb200/gb300/rtx6000/rtx5090/dgx-spark/mi300x/mi355x，**无 Ampere 以下**）；CuTeDSL 内核物理上不支持 SM75 | `docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx:8-18`；`docs/cookbook/.../DeepSeek-V4.mdx` | 硬拦截 |

**旁证**：SGLang 对老卡的底线是 "sm75 and above"（`load_model_utils.py:86`），但那是针对 FP16 传统模型；对 FP8+MXFP4 混合检查点无任何承诺。已验证矩阵最"消费级"的 cell 是 8×RTX 5090（SM120）且 `verified: false`（`deepseek-v4.jsx:2428-2429`）。

---

## 2. llama.cpp 对照：每个阻塞点的可借鉴解法

llama.cpp 侧 DeepSeek-V4 已全链路存在：模型 `src/models/deepseek4.cpp`（1,502 行，含 HC/sinkhorn/compress_ratios/sqrtsoftplus/DSpark）、KV `src/kv-cache-dsv4.cpp`（2,253 行，raw/SWA + 压缩 K-only 块，CSA4/HCA128）、HF 转换 `conversion/deepseek.py:528`（`MODEL_ARCH.DEEPSEEK4`）、量化保护 `src/llama-quant.cpp:308`（tid2eid 路由表不量化）。CUDA 后端对 Turing 是一等公民：`GGML_CUDA_CC_TURING 750`（`ggml-cuda/common.cuh:53`）。

| 阻塞点 | llama.cpp 的解法 | 借鉴到 SGLang 的方式 |
|---|---|---|
| B1/B2（FP8/FP4 门与内核） | **绕开而非硬扛**：GGUF 重量化（Q4_K/IQ4 等 36 种类型），Turing 路径有专用内核 `mmvq.cu:86,108,262,402,1063`（Turing∈[750,860) 的 vec-dot/DP4A 分支）、`mmvf.cu:815`、`mmid.cu`（id 路由 MoE） | Track A：直接转 GGUF 跑通。Track B：给 V4 写 **W8A16/W4A16 的 Triton 反量化 GEMM**（权重保持 FP8/MXFP4 存储，计算期反量化到 FP16），复用 `fused_moe_triton` 的 standard runner 骨架（`fused_marlin_moe.py:302` 已有 capability 分支先例） |
| B3（无 FP8 TC） | 全 FP16/FP32 累加计算，无 FP8 依赖 | wo_a 的 fp8_einsum 分支加 `capability<80 → 反量化 FP16 GEMM` 兜底（`deepseek_v4.py:1831` 的 if 链加一档） |
| B4（DSA 内核） | `llama_kv_cache_dsa` + lightning indexer 融合算子（`llama-context.cpp:59`）；注意力 `fattn-mma-f16.cuh:1821` 有 **Turing 专用 specialization**；`fattn.cu:160,351` 按 cc 分派 | 用 **Triton 写 FP16 paged-MQA logits**（index_head_dim=128、topk=512，规模小，Triton 可达）+ FP16 MLA decode（参考 flashmla 的 Triton 变体）；KV 布局直接照抄 `kv-cache-dsv4.cpp` 的 raw/SWA + 压缩块三段式 |
| B5（后端选择链） | ggml 按 op 选后端 + **CPU 兜底**（`ggml-backend.cpp:1322`）+ meta 后端跨设备切分（`ggml-backend-meta.cpp` 2,511 行） | `attention_registry.py:340-546` 的 if 链加 `(capability<80 → dsa_triton_f16)` 注册项（对齐报告 T9 的注册表方向） |
| B6（无 BF16） | 全栈 FP16/FP32，量化格式自带 scale | 沿用 `maybe_downgrade_dtype_for_legacy_gpu`，但需对 HC/sinkhorn 段做 FP32 累加保护并做数值对拍（vs BF16 参考输出） |
| B7（显存） | `--split-mode layer/row/tensor`（`common/arg.cpp:2842-2848`）+ `--tensor-split` 逐卡配额 + CPU 兜底执行；GGUF mmap 流式加载 | SGLang 已有 `--cpu-offload-gb`/`--offload-group-size`（`server_args.py:3117-3127`，层级卸载）+ `--pp-size`（`:970`）+ `--dcp-size`（`:962`）；配合朋友的双卡 NVLink KV 对半方案（§3） |
| B8（编译） | 本地 CMake 构建 arch 列表含 `75;86`（`llama.cpp-analysis-report.md` §3.2 实测 build cache） | SGLang 需以 `TORCH_CUDA_ARCHS=7.5` 重编 + 把 CuTeDSL/AOT(SM100) 代码路径全部换成可关断言 |

---

## 3. 朋友 fork 的方案评估与整合（TP4+PP2 + NVLink KV 对半）

**他的设计**：8 卡 = TP4 × PP2；KV 按 NVLink 对（2 卡/对）存一份完整副本，对内每卡各存一半，注意力时经 NVLink 读对端一半 → KV 流量锁死在 NVLink（~50 GB/s），不挤 PCIe；跨对（PCIe）只有 TP all-reduce。

**评估**：
- ✅ 正确性：MLA（num_kv_heads=1）的 KV 无法按头切，TP4 下每卡本要存全份；对半方案把 KV 显存 ÷2，PP2 再 ÷2（每 stage 约 22 层），合计 ÷4。V4 压缩 KV 本来就小（128K ctx ≈ 0.5–1 GiB），所以**真正的瓶颈是权重 19.4 GiB/卡**，KV 方案是必要但不充分——必须叠加关 CUDA graph / 限 chunked-prefill / mem-fraction 精调。
- ✅ 拓扑契合：2080Ti NVLink 桥 2 卡/对，KV 读走 NVLink、all-reduce 走 PCIe，方向正确。
- ⚠️ 实现代价：SGLang `mem_cache/memory_pool.py` 假设每 rank 一段连续 KV 池；对半方案需要 peer-access（`cudaIpcOpenMemHandle`/P2P）+ 双段寻址，是 memory_pool 级补丁。
- 💡 待评估变体：**TP2+PP4**（TP 组恰好 = 1 个 NVLink 对）可把 all-reduce 也锁进 NVLink，通信上严格优于 TP4+PP2；代价是 PP4 的流水线气泡。列入 M4 对比实验。

**整合方式**：不重造。M0 拿到朋友 fork 的 diff → 与 `be00a543a7` rebase → 通用部分（capability 门参数化、后端注册表）做成可上游的 patch 序列，私有部分（KV 对半池）留在 fork 分支。

---

## 4. 计划轨道

### Track A — llama.cpp GGUF 路径（短期，目标：这台机器先出字）
- **A1** HF→GGUF 转换：`conversion/deepseek.py`（DEEPSEEK4 已支持），目标量化 Q4_K_M（≈131 GiB）与 IQ4_XS 两档；验证 tid2eid 路由表不被量化（`llama-quant.cpp:308` 逻辑）。
- **A2** 8 卡加载：`--split-mode layer`（保底）与 `--split-mode tensor --tensor-split`（对齐 NVLink 对）两组实验；`-ngl` 从 999 逐步下调找显存/速度拐点。
- **A3** 正确性验收：与 SGLang 参考（H200 上同 prompt 贪心输出）对拍前 64 token 的 logprob 漂移；跑 `perplexity` 对照快照。
- **A4** 性能基线：`llama-bench` 出 pp/tg 数字，作为 Track B 的对照组。

### Track B — SGLang SM75 适配（中期，目标：能跑、可服务）
- **B0** 获取朋友 fork（**阻塞：等用户提供仓库/diff**），rebase 到 `be00a543a7`，跑通其现有路径。
- **B1** 量化门参数化：`SGLANG_ALLOW_SUB80_QUANT=1` 显式放行 + 加载期格式审计（拒绝静默错读）。
- **B2** W4A16 专家 Triton 内核：MXFP4(I8 打包) → FP16 反量化 GEMM，接 standard triton MoE runner；对拍 `fused_marlin_moe` 输出。
- **B3** FP8 注意力 W8A16 兜底：`deepseek_v4.py:1831` if 链加 `<80` 分支（加载期反量化 wo_a 已有流式实现 `_dequant_fp8_wo_a_streaming:4146` 可复用）；fused RMSNorm+FP8 量化退化为 RMSNorm+FP16。
- **B4** DSA Triton FP16 内核：paged-MQA logits（indexer）+ FP16 MLA decode；KV 布局照抄 llama.cpp `kv-cache-dsv4.cpp` 三段式。
- **B5** 后端注册：`attention_registry` 加 capability<80 分支；关闭 CuTeDSL/AOT 路径（改可诊断断言）。
- **B6** 数值对拍：HC/sinkhorn/sqrtsoftplus 在 FP16+FP32 累加下 vs BF16 参考，逐层激活误差 <1e-2。
- **B7** 显存编排：TP4+PP2（朋友方案）与 TP2+PP4（NVLink 全内）对比；关 CUDA graph 起步，稳定后仅对 decode 开小 batch graph；`--chunked-prefill-size 2048` 起步。
- **B8** 构建：`TORCH_CUDA_ARCHS=7.5` 重编内核 wheel；CI 加 sm75 冒烟 job（单机 2 卡即可）。

### Track C — Profiler 探针（横切，支撑 B 的优化决策）
朋友建议的"看 prefill/decode 耗时分布"。**不新造轮子**：SGLang 已有完整基建——
- HTTP `/start_profile`/`/stop_profile`，`ProfileReq` 支持 `profile_by_stage`（prefill/decode 分开）、`profile_stages`、`activities=[CPU,GPU,MEM,RPD]`、`detailed_annotations`（迭代级 KV/请求聚合标注，roofline 用）、`merge_profiles`（`srt/managers/io_struct.py:2117-2143`）；
- NVTX/record_function 标注（`srt/utils/nvtx_utils.py`）+ torch profiler 封装（`srt/utils/profile_utils.py:247-420`）。

**交付物 C1**：`sglang-audit/probe/probe_prefill_decode.py` —— 固定负载（1×4K prefill、64×decode）→ 按 stage 抓 trace → 解析 chrome trace JSON → 输出 markdown 表：每 stage 的 GEMM/Attention/Comm/Memcpy/Schedule 时间占比、Top-10 内核、PCIe vs NVLink 传输量、每 token 解码延迟分解。
**交付物 C2**：阈值规则——decode 中 Comm 占比 >30% → 提示切 TP2+PP4；GEMM 占比 >60% → 提示反量化内核是首要优化点。

---

## 5. 里程碑与验收

| 里程碑 | 内容 | 验收标准 |
|---|---|---|
| M0 | 拿到朋友 fork + 拓扑确认 | fork 可 rebase；目标机 `nvidia-smi topo -m` 确认 4 个 NVLink 对 |
| M1 | Track A 出字 | GGUF Q4_K 在 8×2080Ti 上 `llama-server` 完成 128 token 生成，无 NaN |
| M2 | Track A 验收 | 与参考 logprob 漂移 <5%；bench 基线入库 |
| M3 | B1+B2+B3 合流 | 权重可完整加载进 8 卡（无 OOM），单 prompt 贪心输出与 llama.cpp 版一致率 >95% |
| M4 | B4+B5 合流 | DSA 路径端到端；TP4+PP2 vs TP2+PP4 对比报告（C1 探针产出） |
| M5 | B6+B7 | 128K ctx 下稳定服务 30 min；数值对拍通过 |
| M6 | B8+C2 | sm75 wheel + CI 冒烟；探针报告成为常规回归产物 |

## 6. 风险登记

| 风险 | 等级 | 缓解 |
|---|---|---|
| Triton FP16 内核在 Turing 上性能不达预期（无 async copy、SMEM 小） | 高 | M4 用 C1 探针量化；保底退回 llama.cpp 路径（Track A 已可用） |
| 22G 为魔改 vBIOS 卡，驱动/ECC 行为异常 | 中 | M0 在目标机跑 memtest + `dcgm` 体检 |
| 朋友 fork 与主线漂移过大无法 rebase | 中 | B0 先做 3 文件级试点合并（门参数化/注册表/KV 池） |
| FP16 下 HC/sinkhorn 数值不稳 | 中 | B6 逐层对拍；关键段强制 FP32 |
| CuTeDSL/AOT 内核关不干净 | 低 | B5 全部改可诊断断言 + `SGLANG_DISABLED_*` 开关 |

## 7. 跨会话使用约定

1. **会话开始**：先读 `sglang-sm75-progress.md` 的"状态快照"与"下一步"，再动手。
2. **会话结束前**：更新进度文件的任务状态表 + 追加一行会话日志（日期/会话/改动/证据链接）。
3. 计划文件仅在**里程碑增删或方案变更**时修改，并在文末"计划修订"登记；日常进度只写进度文件。
4. 任务 ID（A1…A4、B0…B8、C1…C2、M0…M6）在两文件间保持一致，不重编号。

## 8. 计划修订记录

| 日期 | 修订 | 理由 |
|---|---|---|
| 2026-09-13 | 初版 | 基于 sglang `be00a543a7` / llama.cpp `73a43d1f6` 实测证据立项；纳入朋友 TP4+PP2+NVLink KV 对半方案与 profiler 探针建议 |
