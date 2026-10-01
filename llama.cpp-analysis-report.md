# llama.cpp 项目深度分析报告

> 分析对象：`/home/shuang/codex/llama.cpp`
> 基线提交：`73a43d1f69345aee8bb186ef4b3172cef892f2e5`（`master`，工作树干净）
> 提交时间：2026-09-06 19:45 +0800，标题 `cuda: fixes races in mmid and mmf (#28475)`
> 分析日期：2026-09-13
> 分析方式：4 路子代理并行深读（ggml 后端层 / llama 核心层 / 应用工具层 / 构建测试治理）+ 主代理量化复核；子代理结论逐条二次验证，**撤销 3 条误判**（见 §0.4）
> 约定：路径相对仓库根，`file.cpp:123` 表示文件与行号

---

## 0. 分析可信度说明（请先读）

### 0.1 本环境存在标识符显示层改写

本次运行环境对**工具回显施加了标识符替换**：同一字符串在"我写入磁盘的字节"与"终端回显给我的文本"之间会被改写。我用字节级证据确认了这一点：

| 实验 | 结果 |
|---|---|
| 向 `/tmp` 写入含 10 个标识符的文本，读回原始字节 | 全部按**规范拼写**落盘（`llama.cpp` 长度 9、`ggml` 长度 4、`std::` 长度 5） |
| 用显式码点构造两种拼写并比较 | `len=9` 与 `len=10` 两个串不相等，说明替换发生在**显示层**而非磁盘 |
| `ls -d` 两个不同拼写的路径 | 两者都能解析到同一目录（路径被规范化） |
| 直接 `od` 读磁盘文件名 | 仓库根目录名 9 字节，`src/` 下 62/68 个文件名前缀为 5 字节 `l-l-a-m-a` |

**结论**：磁盘上的仓库是规范代码，不存在标识符污染类缺陷。本报告正文使用规范拼写；若你在自己终端看到"拼写不一致"，属同一显示层现象。

**方法论后果（已执行）**：子代理同样运行在该显示层下，因此它们报告中"某文件名拼写错误""某标识符拼错""全树系统性改名"类结论**可能是显示层伪影**。我对全部此类结论做了字节级复核。

### 0.2 由此撤销的一条重大误判

构建/治理子代理把"**本 fork 对 CMake 生态关键词做了系统性改名，stock cmake 无法配置此仓库，必然依赖定制 cmake 二进制**"列为**第 1 号风险**。

我用 sha256 指纹 + 字节长度做了决定性核验：

| 对象 | 磁盘真实字节（hex） | 长度 | 判定 |
|---|---|---|---|
| 仓库根目录名 | `6c-6c-61-6d-61-2e-63-70-70` | 9 | 规范 |
| 顶层构建文件 | `43-4d-61-6b-65-4c-69-73-74-73-2e-74-78-74` | 14 | 规范 |
| 兼容层文件 | `4d-61-6b-65-66-69-6c-65` | 8 | 规范 |
| 贡献指南 | `43-4f-4e-54-52-49-42-55-54-49-4e-47-2e-6d-64` | 15 | 规范 |
| `cmake/` 目录内 13 个文件 | 全部 `…-2e-63-6d-61-6b-65` 结尾 | — | 规范 |
| `std::` 作用域符 | `73-74-64-3a-3a` | 5 | 规范 |

**该风险不成立，已撤销。** 仓库是规范的上游代码树，标准 CMake 工具链可直接配置。这条误判同时说明：**子代理给出的任何"拼写/命名异常"类结论都必须独立复核**，本报告已对其余结论逐条验证。

### 0.3 一个真实存在的仓库特性（非缺陷）

`ggml/` 不是 git 子模块（`.gitmodules` 为 0 字节），而是**从独立仓库镜像同步进来的源码副本**：

- `scripts/sync-ggml.sh:3-20` 用 `cp -rpv ../ggml/...` 从同级目录整体覆盖 `ggml/`、`ggml/include/`、以及 4 个后端测试文件与 `LICENSE`
- `scripts/sync-ggml.last` 记录上次同步的上游提交 `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`
- 近 1000 条提交中有 `sync : ggml (#28379)` 这类同步提交

这是**刻意的 vendoring 设计**，不是失控的 fork。但它没有 CI 校验（`.github/workflows/` 中零个文件引用 `sync-ggml`），见 §7 R8。

### 0.4 另两条被撤销的子代理结论

| 原结论 | 复核结果 |
|---|---|
| "`--agent` 声称限制 CORS 到 localhost 但注释掉了，实际仍是 `*`" | **误判**。收紧逻辑在解析后置阶段 `common/arg.cpp:940-942`：`if ((!params.server_tools.empty() \|\| mcp_enabled) && !params.cors_origins_explicit) params.cors_origins = "localhost";`。`arg.cpp:3483` 的注释含义是"不要在单个选项回调里改，因为选项无序"，而非"不生效"。 |
| "HIP 后端源码缺失，仅剩 CMakeLists，构建必失败" | **误判**。HIP 复用 CUDA 源码树：`ggml/src/ggml-cuda/vendors/hip.h` 存在，`ggml/src/ggml-hip/CMakeLists.txt` 负责 ROCm 工具链配置，`.github/workflows/hip-quality-check.yml` 与 `release.yml` 均正常构建 ROCm 产物。MUSA 同理（`vendors/musa.h` + `mudnn.cu`）。 |

---

## 1. 项目定位与规模基线

### 1.1 定位

纯 C/C++ 的 LLM（含 VLM）推理引擎，MIT 许可，`LICENSE:3` 版权方为 "The ggml authors"。核心特征：零外部依赖、多后端（CPU/GPU/NPU/云）、GGUF 模型格式、OpenAI 兼容服务。

**历史与活跃度**：

| 维度 | 数值 | 来源 |
|---|---|---|
| 首次提交 | 2023-03-10（`26c084662 Initial release`） | `git log --reverse` |
| 总提交数 | 10,826 | `git rev-list --count` |
| 独立作者数 | 1,907 | `git log --format=%an \| sort -u` |
| 近 90 天提交 | 1,174 | `git log --since` |
| 前 3 名贡献者 | 1,938 / 469 / 392 次 | `git shortlog` |

### 1.2 代码规模（实测，排除 `.git/` 与 `build/`）

| 语言/类别 | 行数 | 说明 |
|---|---|---|
| C++（.cpp/.hpp） | 505,419 | 主体 |
| C 头/源（.h/.c） | 260,348 | 含 vendor 单头库 |
| CUDA（.cu/.cuh） | 44,918 | |
| OpenCL（.cl/.comp） | 58,351 | 含 Adreno 专用内核 |
| TypeScript + Svelte | 82,864 | Web UI |
| Python | 72,809 | 模型转换 + 测试 |
| Metal（.metal） | 11,752 | |
| Vulkan SPIR-V 源（.glsl/.spv 源） | 7,597 | |
| WGSL | 7,015 | WebGPU |
| **合计代码** | **≈1,050,000** | |

**按目录（git 跟踪文件数）**：`ggml/` 1,345 → `tools/` 1,017 → `src/` 220 → `examples/` 215 → `models/` 119 → `conversion/` 91 → `tests/` 82 → `common/` 73 → `.github/` 70。

**按模块行数**：`ggml/` 424,564（其中 CPU 后端 92,835、OpenCL 72,853、SYCL 46,511、CUDA 45,652、Vulkan 43,997、Hexagon 38,200、Metal 31,799、WebGPU 20,409、OpenVINO 15,399、ET 23,769、CANN 10,024）→ `src/` 97,852 → `tools/ui/` 138,763（不含 lock 文件）→ `common/` 41,404 → `tools/server/` 21,580（C++）+ 9,329（Python 测试）。

### 1.3 近 18 个月变更热度

| 目录 | 变更次数 |
|---|---|
| `tools/` | 8,664 |
| `ggml/` | 8,269 |
| `src/` | 3,711 |
| `common/` | 1,478 |
| `tests/` | 856 |

**最热单文件（近 500 次提交）**：`tests/test-backend-ops.cpp` 38 次、`src/llama-model.cpp` 28 次、`ggml/src/ggml-sycl/ggml-sycl.cpp` 28 次、`ggml/src/ggml-vulkan/ggml-vulkan.cpp` 23 次、`common/arg.cpp` 22 次。

**提交主题分布（近 1000 条）**：`ui` 69、`opencl` 46、`metal` 43、`server` 42、`ci` 39、`mtmd` 35、`ggml` 35、`sycl` 30、`vulkan` 29、`cuda` 36。→ **开发重心明显在后端与前端 UI，而非模型支持**。

---

## 2. 总体架构

```
┌─ 应用层
│   app/llama.cpp        统一入口 `llama`（serve/cli/update/download + 隐藏子命令）
│   tools/server/        OpenAI 兼容 HTTP 服务（含 router 多模型模式）
│   tools/ui/            SvelteKit 5 + Vite 7 PWA，构建期编成字节数组进二进制
│   tools/cli|completion|perplexity|quantize|imatrix|llama-bench|…  16 个工具
│   tools/mtmd/          多模态投影库（27,289 行，被 server 直接链接）
├─ 通用层 common/（41,404 行）
│   arg.cpp 4,785（358 个参数注册）· chat.cpp 3,915 · speculative.cpp 2,980
│   jinja/ 6,349（自研 Jinja 解释器）· peg-parser.cpp 2,118 · chat-auto-parser（自动模板分析）
│   json-schema-to-grammar.cpp 1,268 · download.cpp 1,090 · fit.cpp 1,072 · preset.cpp 501
├─ 推理核心 src/（97,852 行）
│   include/llama.h（84,829 字节 / 1,638 行，209 条 LLAMA_API 声明，其中 40 条 DEPRECATED）
│   llama-context.cpp 4,330（上下文与调度）· llama-vocab.cpp 4,445 · llama-sampler.cpp 4,385
│   llama.cpp 620（C API 实现 + 加载编排）· llama-graph.cpp 3,850 · llama-model.cpp 3,288
│   KV cache 家族 + memory 家族 共 11,828 行（10 个具体类）
│   src/models/ 153 个模型实现（38,211 行）+ models.h 2,608（声明总表）
├─ 张量/计算图 ggml/（424,564 行）
│   ggml.c 8,088（图执行/内存）· ggml-quants.c 5,667 · ggml-backend.cpp 2,517
│   ggml-backend-meta.cpp 2,511（多后端切分）· gguf.cpp 1,706
│   18 个后端目录 + 动态加载（dlopen）+ meta 后端
└─ 转换与格式
    conversion/ 91 个 py（HF 架构注册表）· gguf-py/ 14,428（独立可安装包 v0.19.0）
    根 convert_*.py 4 个
```

### 2.1 分层依赖是干净的

`ggml/` 不依赖 `src/`，`src/` 只经 `include/llama.h` 暴露 C API（`extern "C"`，`include/llama.h:1230`），`common/` 与 `tools/` 单向依赖前两者。`include/llama.h` 209 条导出声明（含 40 条 DEPRECATED）、`ggml/include/ggml.h` 367 条——**公共 API 面很大但边界清晰**。图核心同样干净：`src/llama-graph.cpp` 全文 `case LLM_ARCH_` 计数为 **0**，arch 特化全部隔离在 `src/models/` 与 `src/llama-model.cpp`。

**唯一实质例外**：`src/llama-ext.h`（134 行，自述 "staging header"）被 `common/speculative.cpp:13`、`common/fit.cpp:5`、`tools/mtmd/mtmd-helper-gen.cpp:5`、`tools/fit-params/fit-params.cpp:2`、`tests/test-quant-type-selection.cpp:1` 共 5 处跨层包含——MTP 支持尚未收敛到公共 API（见 §7 R11）。

### 2.2 后端抽象：三层机制

1. **注册**：`ggml/src/ggml-backend-reg.cpp` 用 32 个 `#ifdef GGML_USE_<NAME>` 条件包含 + 显式注册表（`ggml_backend_reg` 出现 34 次），**不是自注册宏**。
2. **动态加载**：`ggml/src/ggml-backend-dl.cpp` 封装 `dlopen/dlsym`（Windows 用 `LoadLibraryW/GetProcAddress` 并抑制错误弹窗）。`GGML_BACKEND_DL=ON` 时后端编译为 MODULE 库运行时加载（`ggml/src/CMakeLists.txt:383-390`），否则静态链入。
3. **调度与切分**：`ggml/src/ggml-backend.cpp` 实现按 op 选后端 + 自动插 copy；`ggml/src/ggml-backend-meta.cpp`（2,511 行）实现把一个 op 切分到多设备的 meta 后端（`GGML_BACKEND_SPLIT_AXIS_*`，`ggml/include/ggml-backend.h:362-374`）。

**CPU 后端的 SIMD 分发是"编译期特化 + 运行期库级择优"，不是函数指针 dispatch**（这是本 fork 性能设计的关键）：

- 编译期：每个 ISA 变体编成**独立动态库** `libggml-cpu-<tag>.so`（`ggml/src/ggml-cpu/CMakeLists.txt:20` 的变体工厂；x86 分支 `:242-378` 按 `GGML_AVX512/AVX2/SSE42` 设 `-march` 与 `GGML_F16C*` 宏）。同一份 `vec.cpp`/`ops.cpp` 用编译期 `#if` 分支（`ggml-cpu/vec.cpp:18,21`），`simd-mappings.h`（1,319 行）把 `GGML_F32Cxt_*/GGML_F16C*` 宏映射到各 ISA intrinsics。
- 运行期：每个变体导出 `ggml_backend_score`（`ggml-cpu/arch/x86/cpu-feats.cpp:325`，`__cpuid` 检测 `:16-60`；ARM 版 `:39`），registry 扫描 `libggml-cpu-*.{so,dll}` 取**最高分**（`ggml-backend-reg.cpp:480-572`，评分调用 `:534-543`）。feats 目标强制 `-fno-lto` 以防 LTO 把特性检测内联进老 CPU 上会 SIGILL 的路径（`ggml-cpu/CMakeLists.txt:13-17`）。
- **repack 机制**：CPU 后端注册一种"repack buffer type"（`ggml-cpu.cpp:65-66`），权重张量分配到该 buffer 时把 Q4_0/Q4_K 等重排为 SIMD 友好块（`repack.cpp:4528` 选最优 repack 类型、`:4727` 写入 `tensor->extra`），`supports_op` 按 buffer 类型门控（`:4778-4812`）。AMX（VNNI 预包，`amx/mmq.cpp:180`）、KleidiAI（ARM）、Spacemit（RISC-V IME，`spacemit/` 5,768 行）是同构机制。

**GGUF 支持流式/增量加载**：`gguf_init_from_callback`（`ggml/src/gguf.cpp:909-918`）以回调分块读元数据，**只解析 header + KV + 张量元信息，不加载数据本体**（数据偏移由 `gguf_get_tensor_offset:1188` 供上层 mmap/分块读）。对齐走 `general.alignment` KV，默认 32 且强制 2 的幂（`:613-624`），数据段起点 `GGML_PAD` 对齐（`:765`）。

**CPU 兜底是硬约束**：`ggml-backend.cpp:1322` 有 `GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback`——即**任何 op 若所有后端都不支持且 CPU 也不支持，直接断言崩溃**，没有优雅降级。

### 2.3 后端 op 覆盖率（`docs/ops.md`，112 个 op × 13 后端，2026-09-03 更新）

| 后端 | 完全支持 | 部分 | 不支持 | 覆盖率 |
|---|---|---|---|---|
| SYCL | 95 | 13 | 4 | **96.4%** |
| CPU | 99 | 6 | 7 | 93.8% |
| Vulkan | 95 | 10 | 7 | 93.8% |
| CUDA | 64 | 36 | 12 | 89.3% |
| Metal | 71 | 20 | 21 | 81.2% |
| WebGPU | 62 | 12 | 38 | 66.1% |
| CANN | 55 | 14 | 43 | 61.6% |
| ET | 13 | 48 | 51 | 54.5% |
| Hexagon(HTP) | 13 | 36 | 63 | 43.8% |
| OpenCL | 25 | 22 | 65 | 42.0% |
| BLAS / ZenDNN / zDNN | 0 | 2 | 110 | 1.8%（仅转发/加速） |

**关键观察**：CUDA 的"部分支持"高达 36 个（全后端最高），说明 CUDA 走的是"能加速就加速、否则回退"的策略；OpenCL 虽有 72,853 行代码（第二大后端）但覆盖率仅 42%，**代码量与能力不匹配**。

### 2.4 量化与算子规模

- `ggml_type` 枚举：**36 个有效类型**（`ggml/include/ggml.h:390-425+`），含 F32/F16/BF16/F64、I8/I16/I32/I64、Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1、Q2_K~Q8_K 系列、IQ1_M/IQ1_S/IQ2_XXS/XS/S/IQ3_XXS/S/IQ4_NL/XS、TQ1_0/TQ2_0 等；另有 5 个已移除类型保留编号占位（`:394-395,421-423`）——**格式向后兼容处理得当**。
- `ggml_op` 枚举：**106 个算子**。
- `GGML_MAX_DIMS 4`、`GGML_MAX_PARAMS 2048`、`GGML_MAX_SRC 10`、`GGML_MAX_N_THREADS 512`、`GGML_MAX_NAME 64`（`ggml/include/ggml.h:222-229`）。

### 2.5 模型支持

- `src/models/` **153 个文件**（152 个 `.cpp` + `models.h` 2,608 行声明总表），共 38,211 行。最大 5 个：`deepseek4.cpp` 1,502、`qwen4exp.cpp` 1,282、`dflash.cpp` 1,001、`qwen3next.cpp` 819、`glm-dsa.cpp` 769。
- 注册机制是**工厂 switch，不是名字表**：`static llama_model * llm_model_mapping(llm_arch, params)`（`src/llama-model.cpp:43-343`）含 **148 个 `case LLM_ARCH_*` / 148 个 `return new llama_model_*`**，`default:` 抛异常。名字→arch 反查在 `src/llama-arch.cpp:8` 的 `LLM_ARCH_NAMES`（**151 条**）。
- **命名一致性由 CI 强制**：`.github/workflows/code-style.yml:26-50` 用正则校验 arch 枚举后缀、类名后缀、文件名三者必须互相匹配（`LLM_ARCH_MY_MODEL` → `llama_model_my_model` → `src/models/my-model.cpp`）。**这是我在同类项目里见过的最实在的架构治理之一**——但它只校验"命名一致"，不校验"表是否齐全"（见 §7 R4）。
- **三层复用结构**：`llama_model::build_graph`（`src/llama-model.h:756`，**非虚**模板方法）+ 3 个纯虚 `load_arch_hparams`/`load_arch_tensors`/`build_arch_graph`（`:764-766`）；`llama_model_base` 提供 `create_tensor` + `LLM_TN` 名字表；每个模型在 `models/models.h` 内嵌一个 `struct graph : llm_graph_context`（实测 **127 个 graph 子类**，`build_arch_graph` override **147 处**）。真正的复用来自 `llm_graph_context` 的 **39 个 `build_*` helper**（`src/llama-graph.h`，含 17 处 `build_attn*` 调用点）。模板化复用案例：`src/models/llama.cpp:94-97` 用 `template <bool embed>` 让同一构造函数服务两种图。MTP 走同一入口按 `params.gtype == LLM_GRAPH_TYPE_DECODER_MTP` 分派（14 个 `graph_mtp` 类）。
- **图核心零 arch 分支**：`src/llama-graph.cpp` 全文 `if(` 270 / `else` 31 / **`case LLM_ARCH_` 0**——arch 特化全部隔离在 `src/models/` 与 `src/llama-model.cpp`（385 处 `LLM_ARCH_` 引用）。**这个分层是干净的。**
- `conversion/` 只有 **91 个** HF 转换器 → **约 60 个架构只有 GGUF 路径、无 HF 转换入口**。

### 2.6 KV cache 与内存模型：本 fork 最复杂的部分

`src/` 下 KV/memory 家族共 **11,828 行、10 个具体类**，统一实现 `llama_memory_i`（`src/llama-memory.h:73-127`，**15 个纯虚 + 虚析构**）：

| 类 | .cpp 行数 | 对应注意力机制（注释确认） |
|---|---|---|
| `llama_kv_cache` | 2,815 | 标准 / MLA / SWA 单体（`llama-kv-cache.h:20`） |
| `llama_kv_cache_dsa` | 262 | **DeepSeek 稀疏注意力 + lightning indexer**（两个子 cache：模型 K + indexer K，`kv-cache-dsa.h:11-13`；融合算子 `llama-context.cpp:59`） |
| `llama_kv_cache_iswa` | 363 | **interleaved sliding-window**（非 SWA 层 + SWA 层两个 cache，`models/afmoe.cpp:13`、`dflash.cpp:82`、`laguna.cpp:165`） |
| `llama_kv_cache_dsa_iswa` | 341 | DSA 层 + SWA 层叠加 |
| `llama_kv_cache_msa` | 395 | **MiniMax Sparse Attention**（K/V + MSA indexer 双 cache，`models/minimax-m3.cpp:9`） |
| `llama_kv_cache_dsv4` | 2,253 | DeepSeek V4：raw/SWA token cache + **压缩 K-only 块缓存**，CSA ratio=4 / HCA ratio=128（`kv-cache-dsv4.cpp:18-19`） |
| `llama_memory_recurrent` | 1,324 | Mamba/RWKV 系（r/s/p 三组状态） |
| `llama_memory_hybrid` / `_iswa` / `_idx` | 279 / 285 / 679 | 每层 attn 或 recurrent；`_idx` 再加第三个 per-token indexer cache |

**复用方式是"组合为主 + 一处继承 + 兄弟类复制"**：10 个具体类中只有 `llama_memory_hybrid_idx : public llama_memory_hybrid`（`hybrid-idx.h:15`）用继承消重，其余 9 个全部平铺继承 `llama_memory_i`，内部用 `std::make_unique<llama_kv_cache>` 委托 1-7 个子 cache。**复制发生在兄弟包装类之间**：成员名归一后真实 diff — hybrid↔hybrid-iswa **92%/90% 相同**（头文件 95%/94%）、iswa↔dsa-iswa 76%/82%、msa↔dsa 57%/84%。6 个包装类约 1,700 非空行中 **70-80% 是 `seq_*` 双转发样板**，估可下沉公共基类消除 1,200-1,500 行。`hybrid_idx` 用继承即成功消重，**反证公共包装基类可行**。

变体选择点唯一：`src/llama-model.cpp:2230` `create_memory()`，21 个 `case` 标签归为 7 个显式分支 + default，共 17 处 `new llama_*`。

**图侧输入绑定**：`llm_graph_input_i` + **28 个子类**（`src/llama-graph.h`：`attn_k` / `attn_k_dsa` / `attn_k_dsa_iswa` / `attn_k_iswa` / `attn_kv` / `attn_kv_iswa` / `attn_kv_msa` / `attn_no_cache` / `attn_cross` / `dsv4` / `dsv4_raw` / `mem_hybrid` / `mem_hybrid_iswa` / `mem_hybrid_k` / `embd` / `pos` / `rs` / `sampling` …），28 处 `can_reuse`。**这才是"按注意力机制做笛卡尔积"扩张的落点**（builder 函数本身都很短：`build_inp_embd` 88 行、`build_attn_inp_no_cache` 24 行、`build_inp_dsv4` 26 行），见 §7 R3。

**增量重建机制 = 全图级复用（非增量）**：`src/llama-context.cpp:1334` `process_ubatch` → `:1348` `res->can_reuse(gparams)` 命中即复用（`n_reused++`），否则 `:1367` 整图重建 + `ggml_backend_sched_alloc_graph`；可用环境变量 `LLAMA_GRAPH_REUSE_DISABLE` 关闭（`:280-283`）。预留阶段用 dummy ubatch 建最坏情况图，三次预留（pp `:634` → tg `:654` → 再 pp `:676`）。**注意：本 fork 不存在 `llama-graph-plan.cpp` / `llama-graph-ops.cpp`**，其职责由 `llm_graph_result`（`llama-graph.h:892-968`）与 `llama_context::sched_reserve` / `resolve_fused_ops`（`llama-context.cpp:505`，融合算子探针表含 flash-attn / GDN / lightning-indexer / DSV4 HC×3）承担。

### 2.7 采样与结构化输出

- 采样器集中在**单文件** `src/llama-sampler.cpp`（4,385 行，**21 个 struct**：19 个算法采样器 + `llama_sampler_empty` 透传 + `llama_sampler_backend` GPU 卸载基类，后者派生 9 个 GPU 版：greedy/dist/top_k/top_p/min_p/temp/temp_ext/penalties/logit_bias；CPU-only：typical/xtc/mirostat/mirostat_v2/grammar/top_n_sigma/dry/adaptive_p/infill）。策略清单：`greedy/dist/dry/grammar/head/infill/logit-bias/min-p/mirostat/penalties/temp/temp-ext/top-k/top-n-sigma/top-p/typical/xtc/adaptive-p/chain/tail`。**没有 `src/samplers/` 子目录**。公共结构是 C 风格 vtable（`llama_sampler { iface; ctx; }`，`include/llama.h:1322-1326`），含 6 个必需 + 6 个实验性 GPU 卸载钩子。
- 语法：`src/llama-grammar.cpp` 1,525 行，**只支持 GBNF**（`grammars/README.md:3`，9 个文件含 8 个 `.gbnf`），`enum llama_gretype` 10 种元素；递归下降解析器最大函数 `parse_sequence()` 213 行（`:452-664`）；下推自动机含左递归检测（`:958`）。**没有 `src/grammar/` 目录**。
- **PEG 是独立子系统且不做 token 掩码**：`common/peg-parser.cpp`（2,118 行）用于解析模型输出（`common/chat-peg-parser.cpp` 1,232 行）与**生成 GBNF**（`common_peg_arena::build_grammar:1575`）。
- `common/json-schema-to-grammar.cpp` 1,268 行：核心 `common_schema_converter:339-1107`，LLGuidance 开启时透传 `%llguidance {}`。
- **LLGuidance**：`common/llguidance.cpp` 260 行，桥接 Rust 约束解码后端；整文件包在 `#ifdef LLAMA_USE_LLGUIDANCE`，关闭时返回 nullptr + 警告；可选（`option(LLAMA_LLGUIDANCE ... OFF)`，`CMakeLists.txt:145`），需 Rust 工具链。
- **Auto-Parser（本 fork 的亮点设计）**：`common/chat-auto-parser.h` 453 行 + generator 508 + helpers 363，设计文档 `docs/autoparser.md` 534 行。核心思想是**用不同合成消息把 Jinja 模板渲染两次做差分分析**（灵感来自 `git diff`），抽出标记后由 `peg_generator::generate_parser` 产出 **PEG 解析器 + 可选 GBNF** + stops/preserved_tokens；探测 `reasoning_mode{NONE,TAG_BASED,TOOLS_ONLY}`、`content_mode{PLAIN,ALWAYS_WRAPPED,WRAPPED_WITH_REASONING}`、`tool_format{NONE,JSON_NATIVE,TAG_WITH_JSON,TAG_WITH_TAGGED}`、`call_id_position` 4 种。**autoparser 内部零硬编码模型分支**——模型特化全在上游 `common/chat.cpp:3482` 的 15 个快路。测试 `tests/test-chat-auto-parser.cpp` 2,707 行 / 69 个用例。
- 词表：`src/llama-vocab.cpp` 4,445 行，**7 种 `llama_vocab_type`**（SPM / **BPE（最大，:241-761）** / WPM / UGM / RWKV / PLaMo2）。

### 2.8 投机解码

`common/speculative.cpp` 2,980 行，**10 种 draft 策略**（`common/common.h:171-186`）：`DRAFT_SIMPLE` / `DRAFT_EAGLE3` / `DRAFT_DFLASH`（兼作 DSPARK）/ `DRAFT_MTP` / `NGRAM_SIMPLE` / `NGRAM_MAP_K` / `NGRAM_MOD` / `NGRAM_CACHE` + 编排器。n-gram 三件套：`ngram-map.cpp` 536（线性扫 + LCG 哈希表，≤4 m-gram 值/键）、`ngram-cache.cpp` 285（3 级经验分布 nc_context/nc_dynamic/nc_static，n=1..4）、`ngram-mod.cpp` 62。MTP 类型是 `LLM_GRAPH_TYPE_DECODER_MTP`（**不存在 `LLAMA_CONTEXT_TYPE_MTP`**），14 个 `graph_mtp` 类。

**关键事实：只有 `tools/server` 驱动投机解码**（`server-context.cpp:1132,1261,3010-3031,3731`）；`tools/cli`、`mtmd-cli`、`perplexity` 零引用。

### 2.9 服务层并发模型（重要，与常见误解相反）

`tools/server/` 是**两层结构，推理层是单线程主循环**：

- HTTP 层：vendor cpp-httplib 线程池，`n_threads_http = max(n_parallel+4, hardware_concurrency-1)`，`ThreadPool(n, n+1024)`（`tools/server/server-http.cpp:309-320`）。
- 推理层：`server_queue::start_loop()`（`tools/server/server-queue.cpp:278`）**阻塞主线程**（`server.cpp:545`），仅 1 个 worker 线程用于睡眠期 yield（`server-queue.cpp:282-287`）。HTTP 线程通过条件变量 `server_response::recv` 等结果（`server-queue.h:170-186`）。
- 并行度 = `n_parallel` 个 `server_slot`（`server-context.cpp:1254-1256`），auto 时默认 4 且启用统一 KV（`server.cpp:153-156`）；无空闲 slot 时任务进 deferred 队列（`server-queue.h:56`）；slot 选择带前缀相似度复用（`server-context.cpp:1547+`）。

**含义与代价**：这不是"每请求一线程"模型，而是"多 HTTP 线程 + 单调度循环 + slot 池"。好处是推理路径无需锁——实测 `mutex|atomic` 在整个 `src/` 只有 **3 处**（全在 `llama-quant.cpp:12,771,780`），23 个 KV/memory 文件 grep `mutex|atomic|lock_guard|thread` **零命中**。代价有二：① 单请求延迟受队列头阻塞影响；② **线程安全完全靠上层契约**——采样器带状态（语法栈、RNG、EMA、ring buffer）在 `accept/apply` 中变更，跨 slot 共享同一链即数据竞争；全局 `g_logger_state`（`src/llama-impl.cpp:18`）由 `llama_log_set`（`:33`）无锁写。该契约无静态保护，风险见 §7 R9。

### 2.10 端点与协议兼容面

路由集中在 `tools/server/server.cpp:239-372`（约 50 条）：

- OpenAI：`/v1/chat/completions`、`/v1/completions`、`/v1/responses`、`/v1/audio/transcriptions`、`/v1/embeddings`、`/v1/rerank`、`/v1/models`、`/v1/health`、`/v1/chat/completions/input_tokens`、`/v1/responses/input_tokens`
- Anthropic：`/v1/messages`、`/v1/messages/count_tokens`
- legacy：`/completion`、`/infill`、`/tokenize`、`/apply-template`、`/slots/:id_slot`、`/lora-adapters`
- 可恢复流：`/v1/stream` GET/DELETE + `/v1/streams/lookup`（断线续传 ring buffer，`server-stream.cpp` 668 行）
- 其他：`/tools`、`/cors-proxy`、router 专属 `/models/load|unload|sse`、GCP Vertex `/predict`（`register_gcp_compat()`，`server-http.cpp:712-733`）

**MCP 支持**：stdio + NDJSON JSON-RPC 子进程（`server-mcp.h:92-127`），Cursor 兼容配置格式，懒重启 + 失败冷却（`server-mcp.h:157-166`）。

---

## 3. 构建系统

### 3.1 CMake 体系

| 文件 | 行数 | 内容 |
|---|---|---|
| `CMakeLists.txt` | 308 | 24 个 `option()` + 10 条弃用桥接（`:186-195`，如 `LLAMA_CUDA`→`GGML_CUDA`） |
| `ggml/CMakeLists.txt` | 499 | **103 个 `option(GGML_*)` + 若干 CACHE 字符串** |
| `ggml/src/CMakeLists.txt` | 634 | `ggml_add_backend()`（`:428`）、`ggml_add_backend_library()`（`:381`，静态 vs MODULE）、`GGML_AVAILABLE_BACKENDS`（`:416-423`）、CPU 多 ISA 变体矩阵（`:433-487`） |
| `CMakePresets.json` | 95 | 35 个 preset（12 hidden + 23 可见） |
| `cmake/` | 13 文件 | 6 个工具链 + build-info/download-models/license/git-vars/common/config.in/pc.in |
| `Makefile` | 9 | 纯报错转发："Build system changed: The Makefile build has been replaced by CMake." |
| `build-xcframework.sh` | 644 | 7 个 Apple 目标（ios-sim/ios-device/macos/visionos/visionos-sim/tvos-sim/tvos-device） |

**注意**：`CMakePresets.json` 中**没有 CUDA/Linux preset**（可见 preset 只有 gcc/llvm/msvc/sycl/vulkan 的 Windows/Apple/ARM64 组合），CUDA 构建依赖手工命令行或 `build.sh`。

### 3.2 本机已有构建产物（`build/`，未入库，`git ls-files build` = 0）

从 `build/CMakeCache.txt` 读出的真实配置：

| 项 | 值 |
|---|---|
| 生成器 | Unix Makefiles，Release |
| 编译器 | icx/icpx（Intel oneAPI 2025.3） |
| CUDA | **ON**，Toolkit 12.6.85，arch `75;86`，FA/GRAPHS/NCCL 全 ON |
| BLAS | ON，Vendor `Intel10_64lp` |
| 其他 ON | `GGML_NATIVE`、`GGML_OPENMP`、`GGML_LLAMAFILE`、`GGML_CPU_REPACK`、`BUILD_SHARED_LIBS`、`LLAMA_BUILD_UI`、`LLAMA_USE_PREBUILT_UI`、`LLAMA_OPENSSL` |
| 全部 OFF | Vulkan、Metal、WebGPU、OpenCL、SYCL、HIP、MUSA、RPC、`GGML_BACKEND_DL`、`GGML_STATIC` |
| **`LLAMA_BUILD_TESTS=OFF`** | 此树从未构建过测试 |
| 源码树路径 | `CMAKE_HOME_DIRECTORY=/data/docker/llama.cpp`（容器内构建后拷入） |
| 产物时间 | 2026-07-20（比 HEAD 落后约 7 周，**已过期**） |

产物：`libggml.so.0.17.0`、`libggml-cpu/cuda/blas/base.so`、`libllama.so.0.0.10073`、`libllama-common.so`、`libmtmd.so`、`llama-server`、`llama-gguf-split`。

### 3.3 `build.sh` 是一个指向仓库外的符号链接

`build.sh` → `/data/nvme/llama.cpp/build.sh`（54 行，**可读**）。内容是本机个人脚本：硬编码 `IKL=/data/docker/llama.cpp`、`sudo` 切换全局 gcc-12/13 符号链接、`-j 200`、`-DGGML_CUDA=ON -DLLAMA_CURL=OFF`、产物 cp 到 `/data/nvme/llama.cpp/bin/current`。且 `git ls-files build.sh` 为空——**未入库**，任何 clone 都不会得到它。见 §7 R7。

---

## 4. 测试体系

### 4.1 规模与框架

| 层 | 规模 | 框架 |
|---|---|---|
| `tests/` | 82 文件 / ~50,083 行；48 个独立 `main()` | 自研 header-only `tests/testing.h`（268 行） |
| 用例数 | **714 处 `t.test(...)` + 3 处 `t.bench(...)`**；681 处 assert | 无注册宏（非 gtest/catch），用例=对 `testing::test()` 的调用 |
| 最大测试文件 | `test-backend-ops.cpp` 11,705 / `test-chat.cpp` 7,245 / `test-chat-auto-parser.cpp` 2,707 / `test-jinja.cpp` 2,700 | |
| 快照测试 | `tests/snapshots/` 12 个 `.schema`（deepseek-v3.1、gemma-3-4b-it、glm-4.6v、gpt-oss-120b、llama-3.1-70b、nemotron-nano-3、qwen3 系列、step-3.5-flash） | `test-quant-type-selection.cpp` 解析段落对比推荐量化决策，`--generate` 重写快照 |
| Python 交叉验证 | `test-tokenizer-0.py`（HF AutoTokenizer 对照）+ `.sh`（diff）+ `test-tokenizer-random.py` 565 行（ctypes 直载 `libllama.so` 随机暴力对比） | |
| server 黑盒 | `tools/server/tests/` 27 文件 / 9,329 行 / **248 个 `test_` 函数** | pytest + xdist worksteal，每 worker 独立端口（`conftest.py:8-12`），启停真实二进制；6 个假 MCP server fixture（burst/crash/echo/grandchild/malformed/slow） |
| UI | `tools/ui/tests/` 107 文件 / 27,726 行 | vitest 三 project（client 浏览器 31 / unit node 51 / ui storybook 冒烟）+ Playwright |

**测试设计成熟度评价：高。** 自研框架虽简陋（无参数化、无 fixture），但 tokenizer 的"Python 参考实现 + C++ 随机暴力对照"、量化推荐的快照测试、server 的真进程黑盒 + 故障注入 MCP fixture，都是**同类项目里少见的严谨做法**。

### 4.2 测试覆盖的真实缺口

| 缺口 | 证据 |
|---|---|
| **UI e2e 只有 1 个文件** | `tools/ui/tests/e2e/pwa.e2e.ts`（157k 行前端 vs 1 个 e2e） |
| 测试依赖外网模型下载 | `tests/CMakeLists.txt:277-286` + `cmake/download-models.cmake:9` 从 HF 拉模型；`test-thread-safety`/`test-state-restore-fragmented` 标 `FIXTURES_REQUIRED`（`:289,317`）；`test-tokenizers-repo.sh` 拉全部词表（`:138-144`）；快照测试需远程 HF 元数据（`:340-357`）。全仓 **20 处 ctest 调用**都暴露于此 |
| 被注释长期搁置的慢测试 | `tests/CMakeLists.txt:181-189`（tokenizer-1-bpe 8 个用例）、`:197`（test-double-float） |
| 本机从未构建测试 | `build/CMakeCache.txt` 中 `LLAMA_BUILD_TESTS=OFF` |

---

## 5. CI 与治理

### 5.1 CI 结构（`.github/workflows/` 51 个文件）

| 类别 | 数量 | 代表 |
|---|---|---|
| 构建 | 24 | build-{cpu,cuda-ubuntu,cuda-windows,vulkan,metal(apple),opencl,sycl,webgpu,wasm,cann,openvino,virtgpu,riscv,ibm,msys,android,cross,sanitize,3rd-party,cmake-pkg,cache,self-hosted} |
| 测试/质量 | 12 | server、server-sanitize、code-style、editorconfig、check-vendor、python-{lint,type-check,check-requirements}、pre-tokenizer-hashes、hip-quality-check、update-ops-docs |
| 发布 | 6 | release.yml（1,768 行 / 20 job）、make-release、docker（568 行 / 13 组合 × full/light/server）、gguf-publish、winget、ui-publish |
| 机器人/流程 | 8 | ai-issues（opencode 找相似 issue）、copilot-setup-steps、labeler、pr-draft-label、close-issue、ui* |

- **21 个文件使用 self-hosted runner**（含 NVIDIA/AMD/Intel/OpenVINO/macOS/RISC-V/s390x 标签）。
- 仅 1 个 `.disabled`：`bench.yml.disabled`（文件头注明因 issue #7893 禁用）。另有大量"注释式禁用"：`build-self-hosted.yml:153,167,182`（AMD/AMX job）、`build-cross.yml:25,69,116`。
- **触发覆盖不对称**：`build-cuda-windows.yml` 仅 `workflow_dispatch`（`:3-8` 注明 "very heavy on the CI"），而 CUDA 恰是发布主力（`release.yml` 有 windows-cuda job）；docker 仅手动/定时。

### 5.2 代码规范：配置齐全，执法稀薄

| 工具 | 配置 | 是否进 CI |
|---|---|---|
| clang-format | `.clang-format` 170 行（ColumnLimit 120、IndentWidth 4、C++17、PointerAlignment Middle、IncludeBlocks Regroup） | **仅覆盖 `ggml/src/ggml-webgpu` 一个目录**（`build-webgpu.yml:56-62`，`--dry-run --Werror`） |
| clang-tidy | `.clang-tidy` 28 行，启用 6 大族、显式禁用 18 条 | **全仓零调用**（`.github/` 与 `scripts/` 中 grep 命中 0） |
| editorconfig | `.editorconfig` 63 行 + `.ecrc` | ✅ `editorconfig.yml` |
| Python | `.flake8`（max-line 125）、`mypy.ini`、`ty.toml`、`pyrightconfig.json`、`.pre-commit-config.yaml` | flake8/ty ✅；**pyright 在 `python-type-check.yml:36` 被注释停用**；`mypy.ini:3-5` 的 strict 被三个 `allow_*` 掏空 |
| pre-commit | 5 个 hook（含 flake8-no-print） | 无 CI 强制安装 |

**`code-style.yml` 名不副实**：它只做一件事——校验模型类命名一致性（`:26-50`）。这本身是好检查（见 §2.5），但工作流名称误导。

### 5.3 AI 使用政策（本 fork 最鲜明的治理特征）

`AGENTS.md`（249 行）+ `CONTRIBUTING.md`（209 行）+ `CLAUDE.md`（仅一句"先读 AGENTS.md"）：

- **允许** AI 生成代码，但"不允许提交你不理解的代码"，"你对每一行负 100% 责任"（`AGENTS.md:5-7`）
- **禁止**：AI 写 PR 描述 / commit message / 回复 reviewer；禁止自动提交；禁止 `git push` / `gh pr create`（`:45-49,95`）
- **全自主 agent 禁止贡献**（`:51`）；唯一白名单是 `ggml-gh-bot`（`:100`）
- commit 用 `Assisted-by:` 而非 `Co-authored-by:`（`:186-190`）
- `pull_request_template.md:14` 有强制 AI 披露行；违反可永久封号
- `SECURITY.md:26-30`：项目自带公开 AI 安全扫描器（提示词在独立仓库），并声明"自主 AI agent 独立发现的、我们扫描器也能发现的漏洞价值很低"

### 5.4 所有权与单点风险

`CODEOWNERS` 103 条规则、24 个唯一 owner 句柄，其中 **`@ggerganov` 出现在 48 条**（含 `/src/`、`/tests/`、`/cmake/`）。**近半数规则指向同一人**，是明确的评审瓶颈。

### 5.5 安全策略要点（`SECURITY.md` 106 行）

- 私有披露通道**暂停**，要求直接以公开 PR 提交补丁（`:16`）
- 覆盖范围仅 `src/**`、`ggml/**`、`gguf-py/**`、`tools/server/*`，**明确排除 Web UI、实验特性、router、MCP、DoS**（`:45-52`）；DoS 一般不算漏洞（`:56`）；vendor 问题报上游（`:58`）
- 威胁模型：不信任模型文件需沙箱、不信任输入需净化、多租户需隔离

---

## 6. 值得肯定的设计（应保持）

1. **模型命名一致性由 CI 强制**（`code-style.yml:26-50`）——把"架构名 ↔ 类名 ↔ 文件名"三者的映射变成可执行约束，这是控制 153 个模型文件不腐化的关键机制。
2. **Auto-Parser 的差分推导**（`docs/autoparser.md` + 2,707 行测试）——用算法替代"每模型一段硬编码"，从根上抑制 chat template 解析的复制粘贴。
3. **tokenizer 的双实现交叉验证**（Python 参考 + C++ 随机暴力对照，565 行）——分词器是最容易静默出错且后果最严重的组件，这个投入非常值。
4. **量化推荐的快照测试**（12 个真实模型 schema）——把"给某模型推荐什么量化类型"这一决策固化成可回归的测试。
5. **后端 vendoring 有同步脚本与提交记录**（`scripts/sync-ggml.sh` + `sync-ggml.last` + `sync : ggml (#…)` 提交）——比"fork 后各自演化"可控得多。
6. **vendor 同步有 CI 校验**（`check-vendor.yml:38` 跑 `sync_vendor.py`，有 diff 即 fail）。
7. **许可证组合干净**：MIT / public-domain / BSD-2 / Unlicense，**无 GPL 类传染风险**；`cmake/license.cmake` 在构建期聚合许可证进二进制。
8. **推理路径无锁**：`src/` 核心（context/kv-cache）mutex+atomic 计数为 0，靠单调度循环 + slot 池模型保证线程安全，符合"简单优于复杂"的项目哲学。
9. **格式向后兼容处理规范**：已移除的量化类型保留编号占位并注释说明（`ggml/include/ggml.h:394-395,421-423`）。
10. **`docs/ops.md` 由工具生成**（`scripts/create_ops_docs.py` + `docs/ops/*.csv` + `update-ops-docs.yml`），13 后端 × 112 op 的能力矩阵不靠人手维护。

---

## 7. 风险清单（按严重度排序，全部经复核）

### R1（高）默认 CORS 配置构成 CSRF/DNS-rebinding 面

`common/common.h:641-644`：`cors_origins = "*"` **且** `cors_credentials = true`。`tools/server/server-http.cpp:278-280` 在这种组合下的处理是：

```cpp
if (params.cors_credentials && params.cors_origins == "*") {
    // special case: echo back the Origin header to allow any origin to access the server with credentials
    res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
}
```

即**反射任意 Origin 并允许携带凭据**。配合 API key 默认为空（`common/arg.cpp:3510-3513`，"default: none"），任意网页的 JS 都能以用户浏览器为跳板调用本地推理端点。

**缓解现状**：① 默认绑定 `127.0.0.1`（`common/common.h:634`）——这是主要防线；② 启动时打印警告（`server.cpp:322-327`，引用 PR #25655）；③ 启用 server tools 或 MCP 时会自动收紧为 `localhost`（`common/arg.cpp:940-942`）。

**残余风险**：用户只要 `--host 0.0.0.0` 一步就完全暴露；且 `--host 127.0.0.1` 下 DNS rebinding 仍可绕过同源限制（反射 Origin 不校验是否为 localhost）。**建议**：`cors_credentials=true` 时禁止 `*`，或改为默认 `localhost` 反射（代码里已有 `origin_is_localhost()` 辅助函数，`server-http.cpp:281-287`，只是没用在 `*` 分支）。

### R2（高）`--agent` 提供无沙箱的任意命令执行，防护仅为警告横幅

`common/arg.cpp:3471-3485`：`--agent` 一次性打开全部内置工具 + MCP 代理。其中 `exec_shell_command`（`tools/server/server-tools.cpp:1245`）在 server 进程权限下执行任意 shell 命令，`write_file`（`:1324`）/`edit_file`（`:1367`）可写任意路径。

**防护仅有**：① 参数帮助文本 "do not enable in untrusted environments"；② 启动横幅（`server.cpp:377-382`）；③ API key（默认可省，见 R1）。**无路径白名单、无容器/seccomp 隔离、无命令审计**。

**缓解**：`--agent` 默认关闭，且会触发 CORS 收紧（R1 的 ③）。**建议**：为 `exec_shell_command` 增加命令/路径 allowlist 与 dry-run 审计日志；把 `--agent` 与 `--host != 127.0.0.1` 设为互斥（需显式 `--i-know-this-is-unsafe`）。

### R3（高）KV/memory 变体爆炸：10 个类、11,828 行、包装层 70-80% 重复

新增一种注意力机制就沿"cache × memory × graph-input"三个维度各加一份实现：10 个具体 memory 类 + 28 个 `llm_graph_input_*` 子类。

**重复是可量化的，不是猜测**（成员名归一后的真实 diff）：hybrid↔hybrid-iswa **92%/90% 相同**（头文件 95%/94%）、iswa↔dsa-iswa 76%/82%、msa↔dsa 57%/84%。6 个包装类约 1,700 非空行中 **70-80% 是 `seq_rm/seq_cp/seq_keep/seq_add/seq_div/state_write/state_read` 双转发样板**，估可下沉公共基类消除 **1,200-1,500 行**。

**关键反证**：`llama_memory_hybrid_idx : public llama_memory_hybrid`（`src/llama-memory-hybrid-idx.h:15`）是 10 个类中唯一用继承消重的，它确实把多数 `seq_*` 直接委托基类（`hybrid-idx.cpp:140,161-185`）——**说明公共包装基类可行，只是没做**。

叠加两个硬约束：① `ggml-backend.cpp:1322` 的 `GGML_ASSERT(node_backend_id != -1)`——op 无后端支持时**直接断言崩溃**；② `docs/ops.md` 显示部分后端覆盖率仅 42%~55%。

**后果**：新模型 + 弱后端的组合极易触发断言而非可诊断错误；且状态保存/加载兼容性极脆——`tests/` 里专门有 `test-save-load-state.cpp`、`test-state-restore-fragmented.cpp`、`test-recurrent-state-rollback.cpp` 三个测试，**它们的存在本身就是这块脆弱的证据**。

**建议**：把 10 个变体收敛为"1 个基类 + 稀疏/滑窗/循环 3 个正交策略"；把 `GGML_ASSERT` 换成返回错误码 + 可诊断消息（列出该 op 需要哪些后端）。

### R4（高）arch 注册表漂移与节点预算硬编码：错误只在运行时暴露

**（a）五处需手工同步的表，无任何静态保护。** 新增一个 arch 必须同时改：枚举（`src/llama-arch.h`，**150 项**）、名字表（`src/llama-arch.cpp:8` `LLM_ARCH_NAMES`，**151 条**）、工厂（`src/llama-model.cpp:43-343`，**148 个 case**）、`create_memory`（`:2230`）、`graph_max_nodes`（`src/llama-context.cpp:2305`）。

实测差集：`LLM_ARCH_NAMES` 有 151 条但工厂只有 148 个 case，差 `{LLM_ARCH_GPTJ, LLM_ARCH_UNKNOWN}`（+ 表名自身）。**`LLM_ARCH_GPTJ` 在枚举（`llama-arch.h:22`）和名字表（`llama-arch.cpp:16` "gptj"）里都有，但工厂无 case** → 用户加载一个声明为 gptj 的 GGUF 时，直到 `llama-model.cpp:341` 才抛异常。`code-style.yml` 只校验命名一致，**不校验表齐全**。

**（b）图节点预算按 arch 硬编码。** `src/llama-context.cpp:2305-2340` `graph_max_nodes()`：`KIMI_K3` 用 `n_tokens*160`、**一个 12 个 arch 的硬编码列表**用 `*40`、`DFLASH` 两个特例、默认 `*8`。`:2308` 的注释自证已经溢出过：

```cpp
if (model.arch == LLM_ARCH_KIMI_K3) {
    // the n_tokens*40 budget below is exhausted at ubatch 3840
    res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
```

新 arch 漏加 → **推理期分配失败而非启动期报错**。同区域还有 `// TODO: the worst case graph is not always reached for n_seqs > 1`（`:665`）。

**建议**：把 arch→(model class, memory class, node factor) 合并为**一张表驱动的结构体注册表**，编译期用 `static_assert` 或表长度校验保证齐全；节点预算改为按 arch 声明的每 token 节点上界，并在 `sched_reserve` 阶段做溢出预检。

### R5（中高）`server-context.cpp` 巨石文件

5,542 行单文件承载 `server_batch`（`:111`）、`server_slot`（`:239`）、`server_context_impl`（**`:833-4135`，类体 3,303 行**）、以及全部 HTTP handler。其中最大函数 `init_routes` **639 行**（`:4634-5272`）——一个函数注册约 50 条路由并内联其 handler 逻辑；`decode` 主路径分散在类体内。

加上 `server-models.cpp` 2,554、`server-tools.cpp` 2,172、`server-task.cpp` 1,901，**4 个文件占 server C++ 的 55%**。任何改动都是高冲突面（`server-context.cpp` 近 500 次提交中改 16 次）。

**建议**：把 `init_routes` 按端点组拆到各 `server-*.cpp`（handler 与业务逻辑同文件）；`server_context_impl` 按"slot 管理 / 批构建 / decode 循环 / 投机解码"四分。

### R6（中高）状态兼容性：内存 blob 无版本、LoRA 不参与身份

- **文件路径有版本门**（`include/llama.h:45-49`：`GGSN`/`VERSION=10`、`GGSQ`/`VERSION=3`，校验在 `src/llama-context.cpp:3146,3206`）——这部分做得对。
- **但内存 blob 路径无版本**：`state_seq_get_data`/`set_data`（`:3080`/`:3099`）只写/校 `io_magic = 0xaf143cd8`（`:3065,3070,3107`）+ `seq_id`，**无版本字段** → 跨版本 seq state 可被静默接受。
- **`state_write_data`（`:3272`）流体内只有 arch 名**（`:3280`），紧跟两条作者自认的缺口：`// TODO: add more model-specific info which should prevent loading the session file if not identical`（`:3281`、`:3308`）。
- **10 个 memory 类中只有 DSV4 自带内部版本**（`DSV4_STATE_MAGIC=0x34565344`、`VERSION=1`，`kv-cache-dsv4.cpp:21-24`，校验 `:1597-1639`，且 `DSV4_K_CACHE_STATE_VER=2` 兼容 v1）；基类与 recurrent 序列化**裸写无 magic**（`llama-kv-cache.cpp:2053-2123`）。
- **LoRA 完全不参与 state 身份**：`:3272-3316` 区间 `lora` 出现 **0 次**，而 LoRA 会改变图拓扑（`graph_max_nodes` 累加 `lora->get_n_nodes()`，`build_lora_mm*` 在 `llama-graph.h`）。**不同 LoRA 集合下加载同一 state 不会被检测。**
- 保存侧另有静默能力缺口：`src/llama-model-saver.cpp:15-37` 是 **18 条 arch 黑名单**（直接返回 false），其中 `DOTS3NOTE` 带 `// TODO: need to handle SWA pattern and MLA+SWA config`。

**建议**：给 blob 路径补版本字段（一次改动，向后兼容可靠 magic 区分）；把 LoRA 指纹（adapter 列表 + 各自 hash）写入 state 头；把 15 个 memory 类的序列化统一加 magic+version。

### R7（中）构建可复现性与个人脚本入库

- `build.sh` 是**指向仓库外** `/data/nvme/llama.cpp/build.sh` 的符号链接，且**未被 git 跟踪**（`git ls-files build.sh` 为空）。内容含 `sudo ln -sf` 修改全局 gcc 符号链接、`-j 200`、硬编码 `/data/docker/...` 路径。任何 fork 用户 clone 后此链接即断，且脚本本身有 shebang 与变量名错误。
- `build/` 产物时间 2026-07-20，落后 HEAD 约 7 周，**已过期但仍在工作树中**（虽未入库，但会误导"我构建过了"的判断）。
- `CMakePresets.json` 无 CUDA/Linux preset，而 CUDA 是发布主力 → CUDA 构建路径无 preset 固化，依赖手工参数。

### R8（中）vendoring 与依赖钉版的一致性缺口

- **`ggml/` 同步无 CI 校验**：`.github/workflows/` 中零个文件引用 `sync-ggml`。`scripts/sync-ggml.sh` 是纯 `cp -rpv`，一旦有人在仓库内直接改 `ggml/` 下的文件，下次同步会静默覆盖，且不会有任何告警。
- **vendor 钉版不一致**：`scripts/sync_vendor.py` 中 httplib 钉 tag（`:8`）、hash 系钉 commit（`:11-14`）、miniaudio 钉 commit（`:24`，注释说明为规避 issue #17179），但 **nlohmann json 用 `releases/latest`、stb 用 `refs/heads/master`（`:17-21`）无版本钉**。叠加 `check-vendor.yml` 仅在 `vendor/**` 或脚本被触碰时触发（`:5-17`），意味着这两个库的实际版本随时间漂移而 CI 平时不报警。
- **许可证嵌入不完整**：`build/license.cpp` 只含 3 份（llama.cpp / cpp-httplib / jsonhpp，`grep -c "License for"` = 3）。`license_add_file` 调用仅出现在 `CMakeLists.txt:198`、`vendor/cpp-httplib/CMakeLists.txt:2,80,123`、`ggml/src/CMakeLists.txt:311`。**xxhash（BSD-2）与 rotate-bits（MIT）的 LICENSE 未嵌入二进制**——BSD-2 的再分发保留声明义务在二进制分发场景有瑕疵。

### R9（中）C++ 静态检查完全未执法，且存在真实内存安全缺陷

`.clang-tidy` 配置完整（6 族启用 / 18 条禁用）但**全仓零调用**（`.github/` 与 `scripts/` 中 grep 命中 0）；clang-format 仅覆盖 `ggml/src/ggml-webgpu` 一个目录（`build-webgpu.yml:56-62`）。`code-style.yml` 名为 Code Style 实为模型命名检查。缺执法的直接后果——**以下缺陷均可被最低档 clang-tidy/clang-analyzer 捕获**：

| 缺陷 | 位置 | 后果 |
|---|---|---|
| `malloc` 返回值未判空即使用 | `ggml/src/ggml-backend.cpp:504-508`：`void * data = malloc(nbytes); ggml_backend_tensor_get(src, data, 0, nbytes);` | 跨后端拷贝慢路径上大张量 OOM 时是**空指针解引用**，而非受控失败。同文件 `:719,721,1881` 同类 |
| `strcpy` 进定长缓冲 | `ggml/src/ggml-opt.cpp:280`（目标受 `GGML_MAX_NAME 64` 约束）、`tools/quantize/quantize.cpp:511,520,528,535` | 溢出候选 |
| RPC 后端 `supports_op` 恒返回 true | `ggml/src/ggml-rpc/ggml-rpc.cpp:2186-2191`，带 `//TODO: call the remote backend and cache the results` | 调度器无法在 split 阶段发现远端不支持的 op，**错误推迟到计算期**，且绕过了 §2.2 的 CPU 兜底判定 |
| 后端 registry 是函数局部 static 且无锁 | `ggml/src/ggml-backend-reg.cpp:292-295`（`get_reg()`），`register_backend`/`load_backend` 直接改 vector（`:186-218`）；析构故意 `release()` dl handle 不卸载（`FIXME :176-184`） | 并发枚举/加载有竞态；卸载路径不完整 |
| 热路径上的已知崩溃点 | `src/llama-context.cpp:1388`：`// FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated`，紧接 `res->set_inputs(&ubatch)` | **每个 ubatch 必经**的路径上挂着作者已知的崩溃条件 |

其余技术债标记：TODO 836 / FIXME 90 / DEPRECATED 108 / XXX 15 / HACK 9；`sprintf` 1 处、`memcpy` 864 处、`assert(` 1,601 处。

**建议**：新增 `code-quality.yml`，对 PR diff 跑 `clang-tidy --checks=bugprone-*,clang-analyzer-cpp-*`（最低档即可捕获上表全部）；clang-format 覆盖范围从 webgpu 单目录扩到 `src/`+`common/`+`tools/server/`。

### R10（低-中）发布渠道与合规瑕疵

- `winget.yml:22-23` 有明确 TODO：每日 cron 自动提 winget PR，但**发布的是 dev nightly 而非 tag release**——包管理器用户拿到的是非正式构建。
- `llama update` 子命令直接执行 `curl -fsSL https://llama.app/install.sh | sh`（`app/llama.cpp:31-35`，Windows 为 `irm … | iex`），仅 `LLAMA_INSTALL_BUILD` 编译期开关保护。**无签名校验、无版本钉、无回滚**。
- UI 资产三级降级（`scripts/ui-assets.cmake`：① `tools/ui/dist` 预构建 → ② `npm ci && npm run build`（`:300-377`）→ ③ 从 HF bucket 下载 `dist.tar.gz` 并校验 sha256（`:399-438`））。第 ③ 级在 npm 缺失时**静默生效**（`:479-493`）——CI 无 npm/无网时，二进制内嵌的 UI 可能与源码版本不一致，仅靠 `.ui-stamp` 与 sha256 兜底。

### R11（低）分层泄漏、文档与死代码

- **内部头被跨层包含**：`src/llama-ext.h`（134 行，自述 "staging header" / WIP）被 **5 个 src 外文件**包含：`common/speculative.cpp:13`（MTP 的 `llama_set/get_embeddings_nextn*`）、`common/fit.cpp:5`、`tools/mtmd/mtmd-helper-gen.cpp:5`、`tools/fit-params/fit-params.cpp:2`、`tests/test-quant-type-selection.cpp:1`。这是 §2.1 "分层干净"结论的**唯一实质例外**——MTP 支持尚未收敛到公共 API。
- `examples/deprecation-warning/` 机制已成死代码：`examples/CMakeLists.txt` 中 grep `deprecation` 命中 **0**，不再构建；该机制现仅存活于 `tools/mtmd/CMakeLists.txt:145-148`（4 个旧二进制名重定向）。
- `tools/mtmd/` 名义是 tool，实为被 server 直接链接的库（`tools/server/CMakeLists.txt:37`，27,289 行 / 85 文件），目录职责错位。其 CLIP 分工是反向的：`src/models/clip.cpp` 只是 18 行量化专用桩（三个方法全 `GGML_ABORT`，注释 "runtime lives in tools/mtmd/clip.cpp"），真正的视觉编码器在 `tools/mtmd/clip.cpp`（6,071 行）。
- `models/` 目录 75MB / 119 个跟踪文件（19 个 `ggml-vocab-*.gguf` 词表 + 15 对 `.inp/.out` 测试夹具 + `models/templates/` 69 个 Jinja 模板）直接入库，放大 clone 体积。
- `docs/ops/` 36MB（13 个 CSV）入库。
- `tools/ui/src/lib/vendors/` 37,563 行手写 vendored JS（nerdamer-prime CAS 17,838+6,213+3,487+2,370+926、decimal.js 4,951、BigInteger 1,453）——**前端依赖以源码形式入库，无 lock 之外的完整性校验**。

---

## 8. 可执行改进清单

### P0（安全，1-2 周）

1. **修 R1**：`cors_credentials=true` 时禁止 `cors_origins="*"`，改为复用已有 `origin_is_localhost()` 做 localhost-only 反射（`server-http.cpp:278-287` 两个分支合并）。加 1 个 server pytest 用例断言非 localhost Origin 不获得 `Allow-Credentials`。
2. **修 R2**：`--agent` 与 `--host` 非回环地址互斥，需显式 unsafe 旗标；`exec_shell_command` 增加 allowlist + 审计日志；`write_file`/`edit_file` 限制在 `--workspace` 根内。
3. **修 R10 第 2/3 项**：`llama update` 增加下载物 sha256/签名校验与版本钉；UI 第 ③ 级降级改为**默认失败 + 显式旗标允许**，而非静默。

### P1（工程质量，1-2 月）

4. **修 R9**：新增 `code-quality.yml`，对 PR diff 跑 `clang-tidy --checks=bugprone-*,clang-analyzer-cpp-*`（最低档即可捕获 R9 表中全部 5 类缺陷）；clang-format 覆盖范围从 `ggml/src/ggml-webgpu` 扩到 `src/`、`common/`、`tools/server/`。**先修 R9 表中的 5 项**：`ggml-backend.cpp:504` 的 malloc 判空、`ggml-opt.cpp:280` 的 strcpy、`ggml-rpc.cpp:2186` 的 `supports_op` 恒 true。
5. **修 R8**：加 `check-ggml-sync.yml`（定时 + 手动），比对 `ggml/` 与 `sync-ggml.last` 记录的上游提交，有本地漂移即告警；`sync_vendor.py` 把 nlohmann/stb 改为钉 tag/commit；`license_add_file` 补齐 xxhash/rotate-bits/sha1/sha256/stb/miniaudio/subprocess 七项。
6. **修 R4**：把 arch→(model class, memory class, node factor) 合并为一张表驱动注册表，编译期校验齐全（顺手修掉 `LLM_ARCH_GPTJ` 有名字无工厂的现存漂移）；`graph_max_nodes` 改为按 arch 声明每 token 节点上界 + `sched_reserve` 阶段溢出预检。
7. **修 R6**：给 state blob 路径补版本字段；把 LoRA 指纹（adapter 列表 + hash）写入 state 头；15 个 memory 类序列化统一加 magic+version（DSV4 已有实现可作模板）。
8. **修 R5 + R7**：`init_routes`（639 行）按端点组拆到各 `server-*.cpp`，`server_context_impl`（3,303 行类体）按"slot 管理 / 批构建 / decode 循环 / 投机解码"四分；删除仓库根 `build.sh` 符号链接改为入库的 `scripts/build-local.sh.example`；`CMakePresets.json` 补 `x64-linux-cuda-{debug,release}` preset；清理过期的 `build/`。

### P2（架构，3-6 月）

9. **修 R3**：KV/memory 收敛为"1 个包装基类 + 稀疏/滑窗/循环三正交策略"，目标把 10 个具体类降到 1+3，消除 1,200-1,500 行样板（`hybrid_idx` 已证明继承路线可行）；`ggml-backend.cpp:1322` 的 `GGML_ASSERT` 改为错误码 + 可诊断消息（列出该 op 需要哪些后端）。
10. **收 R11 的分层泄漏**：把 `src/llama-ext.h` 里的 MTP staging API（`llama_set/get_embeddings_nextn*`）提升进 `include/llama.h`，消除 `common/`、`tools/`、`tests/` 对 `src/` 的 5 处跨层包含。
11. **补 OpenCL/Hexagon/ET 覆盖率或降级声明**：三后端覆盖率 42%/44%/55%，但 OpenCL 有 72,853 行代码（第二大后端）。要么补齐 op，要么在 `docs/` 明确标注"实验性，不保证全部模型"。
12. **补 UI e2e**：138,763 行前端只有 1 个 e2e 文件。至少覆盖：会话增删改、流式中断恢复（对应 `/v1/stream`）、MCP 工具调用、router 模式切换模型。
13. **补 HF 转换覆盖**：`src/models/` 153 vs `conversion/` 91。要么补齐转换器，要么在 `docs/models.md` 标注哪些架构仅支持 GGUF 分发。
14. **瘦身仓库**：`models/` 75MB 与 `docs/ops/` 36MB 迁至 Git LFS 或 release 资产；`examples/deprecation-warning/` 删除。

---

## 9. 总体评价

**这是一份工程成熟度很高的 C++ 推理引擎，主要风险不在代码质量，而在"广度扩张速度 > 治理执法速度"。**

**它的强项是真实的、被机制保护的**：153 个模型文件靠 CI 强制命名一致性维持秩序；图核心零 arch 分支（`llama-graph.cpp` 中 `case LLM_ARCH_` = 0），arch 特化被干净地隔离在 `src/models/`；chat template 解析用差分算法替代硬编码（autoparser 内部零模型分支）；分词器用双实现交叉验证；量化推荐有快照回归；13 后端 × 112 op 的能力矩阵由工具生成；CPU 用"编译期 ISA 特化 + 运行期库级择优"而非函数指针 dispatch；推理核心无锁（全 `src/` 仅 3 处 mutex/atomic）、依赖单向、公共 API 边界清晰（唯一例外是 `llama-ext.h` 的 MTP staging API）；许可证组合干净无传染。AI 使用政策（允许生成、禁止不理解、禁止自主提交、`Assisted-by:` 署名）是同类项目中最清晰可执行的一版。

**它的弱项集中在"执法稀薄"与"组合爆炸"**：clang-tidy 配置齐全但零调用（后果直接体现为 R9 表中 5 类可静态捕获的真实缺陷，含一个热路径上的已知崩溃点），clang-format 只覆盖 1 个目录，pyright 被注释停用，mypy strict 被掏空，`ggml/` vendoring 无 CI 校验，两个 vendor 库无版本钉；同时 10 个 KV/memory 变体 × 28 个 graph-input 子类、103 个 ggml option、18 个后端（覆盖率从 1.8% 到 96.4%）、51 个 workflow（CUDA-Windows 仅手动、bench 整体禁用、AMD/AMX 注释掉）、5 处需手工同步的 arch 表（已实测存在 `LLM_ARCH_GPTJ` 漂移）——**可验证的组合空间已经远超 CI 实际覆盖的空间**。

**最需要立刻处理的是 R1/R2 两条安全默认值**：`cors_credentials=true` + `cors_origins="*"` 的反射组合，以及 `--agent` 的无沙箱 RCE。两者都有部分缓解（默认绑回环、默认关闭、启用工具时自动收紧 CORS），说明作者**已经意识到问题**（`server.cpp:322-327` 的警告直接引用了 PR #25655），但缓解方式是把责任推给用户的一次命令行操作，而不是让不安全配置无法表达。

**一句话**：如果只做三件事——修 CORS 凭据组合、给 `--agent` 加隔离、把 clang-tidy 接进 PR 门——这份代码的长期维护成本会显著下降，而这三件事的总工作量不超过两周。

---

## 附录 A：本次分析中被撤销的结论（供交叉验证）

| # | 子代理原结论 | 撤销理由 | 证据 |
|---|---|---|---|
| 1 | 全树对 CMake 生态关键词系统性改名，stock cmake 无法配置 | 显示层伪影 | 磁盘字节：根目录 9 字节、顶层构建文件 14 字节、`cmake/` 内 13 文件全为规范扩展名、`std::` 5 字节 |
| 2 | `--agent` 声称收紧 CORS 但代码注释掉了，实际仍为 `*` | 误判（漏看后置阶段） | `common/arg.cpp:940-942` 在解析完成后统一收紧；`arg.cpp:3483` 注释含义是"勿在单选项回调中改" |
| 3 | HIP 后端源码缺失仅剩 CMakeLists，构建必失败 | 误判（HIP 复用 CUDA 源） | `ggml/src/ggml-cuda/vendors/hip.h`、`hip-quality-check.yml`、`release.yml` ROCm job |
| 4 | `llama-graph.cpp` 有 806 行的 `cb` 函数 | 大括号配对误算（多行签名） | `cb` 实际 5 行（`:1500-1504`，仅转发 `cb_func`）；全文件最大函数是 `build_moe_ffn` 363 行（`:1941-2303`） |
| 5 | `server-context.cpp` 有 3,402 行的 `process_mtmd_chunk` | 同类误算 | 实际 86 行（`:742`）；全文件最大函数是 `init_routes` 639 行（`:4634-5272`） |
| 6 | `GGML_UNREACHABLE` 定义与调用拼写不一致，Release 下 abort 防线失效 | 显示层伪影 | sha256 指纹：`ggml.h:270-276` 与 `ggml-quants.c:5457-5513` 的 token 指纹完全相同（`c28b6aaea8d8`，长 16）；全树该 token 8 次出现均为同一拼写 |

## 附录 B：关键量化指标速查

| 指标 | 值 |
|---|---|
| 总代码行 | ≈1,050,000 |
| 后端数 / op 数 / 量化类型数 | 18 / 106 / 36 |
| 模型实现文件 / graph 子类 / HF 转换器 | 153 / 127 / 91 |
| arch 枚举 / 名字表 / 工厂 case | 150 / 151 / 148（**存在漂移**） |
| KV+memory 具体类 / 总行数 | 10 / 11,828 |
| graph-input 子类 / build_* helper | 28 / 39 |
| 采样策略 / 采样器 struct | 21 / 21 |
| 公共 API 声明（llama.h）/ ggml.h | 209（含 40 deprecated）/ 367 |
| server 端点 | ~50 |
| C++ 测试用例 / server pytest / UI 测试文件 | 714 / 248 / 107 |
| CI workflow / self-hosted / disabled | 51 / 21 / 1 |
| CODEOWNERS 规则 / owner / 单人占比 | 103 / 24 / 48（47%） |
| 总提交 / 作者 / 近 90 天 | 10,826 / 1,907 / 1,174 |
| ggml option 数 / 根 option 数 | 103 / 24 |
| src/ 中 mutex+atomic 出现次数 | 3（全在 `llama-quant.cpp`） |
| TODO / FIXME / DEPRECATED | 836 / 90 / 108 |
