# 生产状态 — DeepSeek-V4-Flash on 8× RTX 2080 Ti (SM75)

**这份文件记录「盒子上此刻实际在跑什么」**,与 `PROGRESS.md` 的区别:那份记的是已发布
里程碑(每个对应一个 GitHub release),而当前状态**尚未打包成 wheel、尚未打 tag**,
只存在于两个仓库的分支上。

最后更新:2026-10-02 20:30 · 全部数字为当次实测,非估算

---

## 1. 跑的是什么

| 项 | 值 |
|---|---|
| 服务 | `/data/nvme/sglang/deepseek-v4-flash.sh`,端口 8200,日志 `logs/serve-prod.log` |
| 模型 | `/data/nvme/models/DeepSeek/DeepSeek-V4-Flash-Vision-Exp`(fp8 e4m3,block 128×128) |
| 并行 | **TP2 × PP4** |
| 上下文 | **262,144**(256K) |
| KV 池 | `max_total_num_tokens=264192`,fp8_e4m3,启动时 `available_gpu_mem` 0.34 GB |
| 层级分派 | `SGLANG_PP_LAYER_PARTITION=11,11,11,10`(必须 `export`,见下) |
| chunked prefill | **256** |
| mem-fraction-static | 0.97 |
| decode graph 桶 | `4096,8192,16384,32768,65536,131072` |
| 功耗上限 | 150 W/卡(经验证的上限,更高有 GPU6/7 挂死的风险) |

### 代码对应

| 仓库 | 分支 / commit | 内容 |
|---|---|---|
| `p4s2wd/sglang-sm75` | `sm75-dsv4-flash-main` @ `253d3fae31` | 框架改动:并行 top-K、IMA 修复、开关与诊断 |
| `p4s2wd/sglang-dev` | `main` @ `ee54abd` | 本文件、基准脚本、NOTES、`launcher/` 受控副本 |

**sglang 侧今天两个 commit:**

- `551da196b1` — decode 的稀疏 top-K 并行化(bs=1,256K 上下文 **+41.8%**)
- `253d3fae31` — `SGLANG_OPT_SM75_PARALLEL_TOPK=0`  kill switch +
  `SGLANG_OPT_SM75_TOPK_DIAG=1`  一次性 trace

**launcher 侧一个改动:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

---

## 2. 实测性能

测量口径:prefill 用 `pf_logprobe.py`(prompt 里插随机 nonce 破坏 radix 前缀,
`prompt_tokens / wall`);decode 用 `dec_ctx_logprobe.py`(读调度器自报的
`gen throughput`,取中位数,并剔除 prefill 边界造成的假采样)。

### Prefill

| prompt token | 墙钟 | tok/s |
|---|---|---|
3,987 | 2.8 s | 1402 |
32,001 | 20.2 s | 1585 |
128,003 | 121.0 s | 1058 |
255,999 | 408.3 s | **627** |

短 prompt 是这条曲线的最高点(32K 时 1585),到 256K 掉到 627(−60%)。
**用户视角:一个 256K prompt 要等约 7 分钟才出第一个字。**

### Decode(bs=1)

| 上下文 | tok/s | 今天改动前 | 变化 |
|---|---|---|---|
8,448 | 26.86 | 27.73 | −3% |
32,256 | 25.99 | 26.35 | −1% |
65,792 | 23.85 | 22.39 | +6.5% |
131,328 | 21.48 | 18.07 | **+18.9%** |
155,904 | 21.31 | 17.11 | **+24.6%** |
200,960 | 21.12 | 16.08 | **+31.3%** |
**260,352** | **20.69** | **14.59** | **+41.8%** |

从 8K 到 256K,原来掉 47%,现在掉 23%;**131K 之后基本平**(21.48 → 20.69,−3.7%)。

### 并发(wall-clock 口径,与上表口径不同,勿混比)

| bs | 聚合 tok/s |
|---|---|
1 | 24.62 |
2 | 42.19 |
4 | 70.85 |
8 | **113.11** |

---

## 3. 这些数字在什么位置

模型 93.8% 的权重是 MoE 专家,每 token 只激活 6/256 个(2.34%)。按 safetensors 头部
实测字节:

| | 数值 |
|---|---|
每 token 激活权重 | 12.81 GB(全模型) |
每卡 | 1.60 GB |
负载下实测显存频率 | 6800 MHz(≈满速),SM 1485–1575 MHz(86–91% boost) |
**bs=1 理论上限** | **~370 tok/s** |
**现在(bs=1, 短上下文)** | ~25–27 tok/s = **上限的 7%** |
**现在(bs=1, 256K)** | ~21 tok/s = **上限的 5.6%** |

**离天花板还有约 14 倍。** 差距全在主机侧 —— 30 秒 py-spy profile(bs=1, 8K):

| 主机侧在做什么 | 占比 |
|---|---|
CUDA graph replay(真正的计算) | 16% |
**PP 收发**(`recv_tensor_dict` → pickle 走 CPU gloo TCP) | 14–25% |
**KV 分配**(`alloc_for_decode` 等) | ~19% |
TP 集合通信(all_gather / broadcast) | ~8% |
GPU 忙的时间占比 | **4.8%**(kernel 时长中位数 4 us) |

**PP4 在 bs=1 下零流水收益**:只有 1 个 token 在飞,4 级时间直接相加,却要付 3 次
主机端阻塞握手。加 bs 也救不了 —— 同一批的多个 token 一起穿过 4 级,还是串行;
bs=8 的提升来自权重只读一遍,不是流水。

**这是下一步(B)的目标。**

---

## 4. 今天落地的两项改动

### 4.1 decode 稀疏 top-K 并行化(`551da196b1`)

生产 kernel 的 grid 恒为 `[1,1,1]`,一个 program 只用一个 SM(68 个里的 1 个)。
拆成分片并行 + 全排序归并。

| 宽度 | 原 kernel | 并行版 | 加速 |
|---|---|---|---|
8,192 | 202 us | 122 us | 1.7× |
37,500 | 671 us | 111 us | 6.0× |
150,000 | 2,908 us | 141 us | 20.6× |
262,144 | 4,525 us | 177 us | **25.6×** |

阈值全部实测:宽度 < 8192 或 rows > 16 时走原 kernel(小 rows 时原 kernel 本来就有
并行度,拆开反而更慢)。

**踩过的坑,以及它为什么值得记下来:** packed key 是 `(float_bits << 32) | 列号`,
而正浮点的 `float_bits ^ 0x80000000` 会到 `0xFFFFFFFF` —— 左移后 bit63 置位,
**一半左右的真实分数作为 int64 是负数**。有符号排序会把最大的分数排到最后,
零填充再占满输出槽。生产 kernel 的 `acc` 是 `uint64` 所以一直是对的,是我的原型
存成了 int64。更阴险的是 `torch.sort` 对 int64 也是有符号的,拿它当真值时
**错实现和错答案给出了相同顺序**。「生产 merge 不精确」那条结论整个是伪像。

### 4.2 PP0 prefill OOM 修复(`fc2c814`,launcher)

```
Tried to allocate 44.00 MiB ... 35.25 MiB is free
... 119.90 MiB is reserved by PyTorch but unallocated
```

**内存是够的,碎成了小于 44 MiB 的块。** 那 44 MiB 是 MoE 暂存
(`num_slots = num_tokens×topk + (E+1)×(block_m−1)`),不是索引器。

`expandable_segments:True` 之后:

- 冷 prefill **6/6 通过**(256,933–259,908 token),其中 3 次是在 6 分钟混合负载
  把碎片攒起来之后压的 —— 原始崩溃就发生在那种状态下。改之前同类请求崩过 1 次。
- prefill 吞吐 **+3~6%**(不是变慢)。
- decode 也快了:bs=1 **+12%**、bs=4 **+31%**、bs=8 +2%。碎片少了,分配器就不用
  每步去 cudaMalloc,这与 `alloc_for_decode` 占 19% CPU 时间对得上。

---

## 5. 已知限制(必须一起读)

| 项 | 状态 |
|---|---|
| **KV 池 267,776 → 264,192** | `expandable_segments` 预留更大虚拟段的代价。仍 > 262,144,但 256K 的余量从 5,632 缩到 **2,048 token** |
| **greedy parity 5/6,不是 6/6** | prompt[2] 在第 129 字符分叉,两版都通顺、答同一件事。同配置两次抓取逐字节一致 → 确定性没坏,是分配器换了地址导致 kernel 选路不同 → 一个 argmax 翻。GSM8K 严格口径反而 0.890 → **0.910**。**数值良性,但可观测行为确实变了,不能说「和之前一样」** |
| **256K 可靠性样本只有 6 次** | 6 过 1 崩(崩的那次在修之前)。只能说「不再立刻崩」,不够称可靠 |
| **短上下文 decode 有 1–3% 回落** | 8K / 32K 在测量散布内。那个区间并行路径不启用(阈值 8192),更可能是噪声,但不是零 |
| **`chunked-prefill-size` 不能设 512** | 实测崩服:PP0 要 56.00 MiB、只剩 55.25 MiB,**差 0.75 MiB**。崩在 ~139K 上下文、52% 池占用 |
| **只测了 bs=1 和并发** | 并发曲线的收益是否被分摊掉,没有系统测过 |

---

## 6. 操作

```bash
cd /data/nvme/sglang

./deepseek-v4-flash.sh              # 起
./deepseek-v4-flash.sh --stop       # 停
./deepseek-v4-flash.sh --check      # 只做环境自检

# 崩溃后先归档日志再重启(规矩)
cp logs/serve-prod.log /data/nvme/sglang-codex/plan2026-09-30/res/crash_<label>.log
```

**serve 日志会轮转,不要用 `grep 'fired up and ready'` 判断就绪** —— 旧启动的行可能还在。
按启动前的行数定位:

```bash
N0=$(wc -l < logs/serve-prod.log)   # 启动前记下
./deepseek-v4-flash.sh &
tail -n +$N0 logs/serve-prod.log | grep -q "fired up and ready"
```

**所有配置都是环境变量覆盖**,不用改文件: `TP` / `PP` / `CHUNK` / `MEM_FRACTION` /
`PYTORCH_CUDA_ALLOC_CONF` / `SGLANG_PP_LAYER_PARTITION`。

### 开关

| 变量 | 默认 | 作用 |
|---|---|---|
| `SGLANG_OPT_SM75_PARALLEL_TOPK` | 1 | 设 0 退回单程序 top-K kernel(A/B 用) |
| `SGLANG_OPT_SM75_TOPK_DIAG` | 0 | 设 1 打印每个 shape 走哪条路(只打一次) |
| `SGLANG_TRITON_SYNC_EVERY_LAUNCH` | 0 | 每个 Triton launch 后同步,用于把 IMA 报告限制在「晚一个 launch」 |

**注意:`/proc/<pid>/environ` 对 scheduler 进程不可信** —— 它们 `setproctitle` 过,
报的是 fork 时的环境。判断 worker 实际跑了什么,只能看它自己的日志。

### 回退

| 想退回 | 做法 |
|---|---|
| 并行 top-K | `SGLANG_OPT_SM75_PARALLEL_TOPK=0 ./deepseek-v4-flash.sh` |
| expandable_segments | `PYTORCH_CUDA_ALLOC_CONF= ./deepseek-v4-flash.sh`(会回到 267,776 池,但 256K prefill 可能 OOM) |
| chunk 512 | **不要。** 会 OOM 崩服 |
| 框架代码 | fork 的 `sm75-dsv4-flash-main` 分支,前一个 commit 是 `2fb2c126ce`。**不要用 `git checkout` 回退生产代码** —— 那正是 10-01 搞崩服务的手法,用精确编辑 |

---

## 7. 明确没做的

- **`block_m` 16→8**:读过代码后判定**不该做**。prefill 时 align 缓冲的浪费只有 24%
  (容量 5,391 行 vs 需要 ~4,096),96% 的浪费在 decode,而 decode 的缓冲被 CUDA graph
  缓存复用、只驻留一份,不是峰值问题。省不到 24% 却要改 GEMM tiling,不划算。
- **PP4 → PP2×TP4**:未测。每卡权重不变(都是总量/8),PP 跳数 3→1,但 bs=1 下每级要扛
  2 倍层,预计只有 27→31 量级。风险是 PP0 从 11 层变 22 层,260K prefill 峰值可能翻倍。
- **动 `recv_tensor_dict` 的 pickle 阻塞路径**:这才是大头(占 25%),但侵入式。
- **ncu/nsys 级 profiling**:torch profiler 会把吞吐压到 1/10 并严重扭曲 NCCL,
  现有结论都避开了它。

---

## 8. 详细记录

| 文档 | 内容 |
|---|---|
| `plan2026-09-30/NOTES-2026-10-02-decode-topk.md` | top-K 调查全过程,含 4 个被推翻的结论和它们为什么错 |
| `plan2026-09-30/NOTES-2026-10-02-longctx.md` | 256K 上下文与 SWA 计价修复,含 5 次操作失误 |
| `PROGRESS.md` | 已发布里程碑历史 |
| `release/sm75main2/RELEASE_NOTES.md` | 上一个正式 release 的说明 |

测量脚本都在 `plan2026-09-30/`:`pf_logprobe.py`(prefill)、
`dec_ctx_logprobe.py`(decode 曲线)、`dec_batch_scale.py`(并发)、
`pf256k_stress.py`(碎片后压 256K)、`gsm8k_gate.py`、`greedy.py`(parity)、
`topk_prod_validate.py`(top-K 单元验证)。

**每个脚本的 docstring 都写了它为什么那样测** —— 大部分是踩过坑之后才写对的
(比如 radix 缓存会把 prefill 吃掉、参考系本身会被同一个 bug 污染)。