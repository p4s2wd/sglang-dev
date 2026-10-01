# 2026-10-02 夜间:长上下文 decode 与 256K 上下文

接 `plan2026-09-30/NOTES.md`。本篇是独立记录,因为它推翻了两个"已关闭"的结论。

## 一、问题的重新定性:不是 decode 慢,是上下文根本用不到配置的长度

用户报的现象是"上下文变长时 decode 掉到 12-18 tok/s"。实测 bs=1 单请求的 decode 曲线
(`plan2026-09-30/dec_ctx_probe.py`,缓存命中相减法):

| 上下文 | ms/token | tok/s |
|---|---|---|
| 2,891 | 36.57 | 27.35 |
| 11,559 | 37.70 | 26.53 |
| 46,143 | 41.56 | 24.06 |
| 92,128 | 46.11 | 21.69 |
| 119,467 | 49.16 | 20.34 |
| 156,429 | 60.67 | 16.48 |

**上下文涨 41 倍,decode 只慢 26-40%** —— 斜率温和,attention 不是主因。

真正的原因:

```
Input length (184056 tokens) exceeds the maximum allowed length (162298 tokens)
```

`--context-length 262144` 是**虚的**。实际 KV 池只有 **162,304 token**,超了直接 400。
而一旦逼近上限,KV 被驱逐,新 turn 重算前缀 —— 15 万 token 的 prefill 按 1300 tok/s 算要
**130 秒**。**用户感觉到的"decode 变慢",实际是 prefill 在反复重跑。**

143K 那一档测出 285 tok/s 的物理不可能值,正是池满的信号:第二次调用触发驱逐+重算,
把"两次相减"的方法算崩了。

## 二、decode 增长的真正来源:`topk` 选择,不是 sparse attention

`SM75_DSV4_DECODE_PROFILE.md` 判定"attention 已调优、无空间",依据是短上下文下的
census(`headshared_sparse` 0.96 ms / 12.94 ms)。**那个结论在长上下文下依然成立。**

同一 census 在 440 token vs 150K token(PP3,`plan2026-09-30/dec_census.py`):

| kernel 家族 | 短 | 长 | 增长 |
|---|---|---|---|
| **`_topk_transform_paged_triton_kernel`** | 1,497 us | **51,600 us** | **34.5x** |
| **`_mqa_paged_smallq_kernel`** | 3,095 us | 31,093 us | 10.0x |
| `attn_sparse(MLA)` | 10,325 us | 19,189 us | 1.9x |
| gemm | 34,555 us | 32,612 us | **0.94x** |
| moe | 18,715 us | 17,350 us | **0.93x** |

**gemm 与 moe 完全不随长度变(0.94x / 0.93x)**,说明模型计算部分与上下文无关;增长全部
来自"选择哪几个 KV 页"。`attn_sparse(MLA)` 占比反而从 6.0% 掉到 5.1%。

**下一个优化目标已经锁定:`_topk_transform_paged_triton_kernel`**(即 2026-10-01 修 IMA 的
那个 kernel),长上下文下是 PP3 上最大的非通信 kernel,2.15 ms/token。

## 三、`mem_fraction_static` 往上没人试过

9/29 只试过往下调(0.94 被拒,硬下限 0.9548)。往上:

| mem-fraction | KV 池 | PP0 余量(F3) |
|---|---|---|
| 0.97 | 162,304 | 0.64 GB |
| **0.98** | **269,824** | 0.46 GB |

0.98 让池 +66%,**第一次超过模型自己配置的 262,144**。但 260K prefill 与 200 题 GSM8K
都会崩(`ncclAllGather` 处 unhandled cuda error,与 `PREFILL_PROFILE.md:637` 记的 9/29
崩溃一字不差)。崩溃时 KV 池使用率只有 0.01 —— **不是池耗尽,是 PP0 余量撑不住 NCCL。**

## 四、F3 与 256K 的冲突是结构性的

F3(`11,11,11,10`)的 +7.6% 来自"PP3 每层最贵(25.0 ms,带 lm_head),把层从 PP3 挪走"。
各 stage 每层成本:PP0 **21.5** / PP1 23.2 / PP2 22.5 / PP3 **25.0** ms。

**要加速 prefill 就得把层搬到最便宜的 PP0,而 PP0 正是唯一装不下 256K 的 stage:**

| | PP0 权重 | PP0 余量(0.98) | 池 |
|---|---|---|---|
| `10,11,11,11` | 18.44 GB | 0.94 GB | 269,824 |
| `11,11,11,10` | 20.14 GB | 0.46 GB | 269,824 |

降 `chunked_prefill_size`(512→256)能让 260K prefill 通过(prefill 吞吐基本中性:短 prompt
+16%、长 prompt −10%),但 **GSM8K 照崩**。

## 五、★ 根因:SWA 显存被高估了近 5 倍

`pool_configurator.py` 的 `bytes_per_full_token` 分解(SWA 修复前,PP0):

| 组成 | 字节/token | 占比 |
|---|---|---|
| SWA | 835.50 | 43% |
| c4 层(6 × 179.25) | 1075.50 | 55% |
| c128 层(5 × 6.75) | 33.75 | 1.7% |
| **合计** | **1944.75** | |

而 SWA 层的实际分布(`stage_compress_ratios = cfg.compress_ratios[stage.start:stage.stop]`):

| stage | local_layers | ratios | 实际 SWA 层 | 被收的 SWA 费 |
|---|---|---|---|---|
| **PP0** | 11 | `[4,4,4,4,4,4,128,128,128,128,128]` | **0** | **835.50** ← 纯浪费 |
| PP1 | 11 | `[0,0,4,4,4,4,4,128,128,128,128]` | **2** | 803.50 |
| PP2 | 11 | `[4,4,4,4,4,128,128,128,128,128,128]` | 0 | 803.50 |
| PP3 | 10 | `[4,4,4,4,4,128,128,128,128,128]` | 0 | 745.00 |

**全局只有第 0、1 层是 SWA,两层都在 PP1。PP0/PP2/PP3 的 swa pool 是空的,却按自己
全部层数付费。** 原代码 `_get_bytes_per_swa_token` 用 `num_layers_total`,已改为只对
`stage_compress_ratios` 里等于 0 的层计费(`swa_layers == 0` 时返回 0)。

c4 state 项保持原样 —— 它按 c4 层计价,而每个 stage 都有 c4 层。

## 六、被硬件代际挡死的两条路

| 想法 | 阻断原因 |
|---|---|
| `--enable-deepseek-v4-fp4-indexer`(索引器 132→68 B,省 48%) | `serving_hook.py:423` 硬性要求 SM100/SM120/gfx95,本机 SM75 |
| EAGLE / MTP 投机解码 | draft 权重 5.34 GiB/卡,总权重 164.91 GB / 容量 176 GB = 93.7% |

## 七、KV 放主机内存:容量上可行,性能上不看好

主机 503 GB(可用 377 GB)vs 显存 KV 池 0.32 GB,差 1000 倍。HiCache
(`--enable-hierarchical-cache` + `--hicache-ratio` / `--hicache-size`)是内置方案,
且形状上对:DSV4 稀疏注意力每步只读 O(topk)。

**但 PCIe 是瓶颈**:从 profile 反推 PP3 每步读约 53 MB KV,本机 PCIe 实测 10.3 GB/s
→ 5.1 ms/step,比现在 0.80 ms 慢 6 倍。**未验证 HiCache 是否支持 DSV4 的四段式池布局
(full/swa/c4/c128,非扁平页池)。**

## 八、本轮我犯的操作错误(5 次,全部在生产上)

按严重性排序,前两类已加机制拦截:

1. **在压测期间启动 restart** → 杀掉被测服务,得出**假的崩溃结论**。已加机制:
   `ab_restart.sh` 检测到 `:8200` 有健康服务就 `exit 2` 拒绝执行,需显式 `AB_FORCE=1`。
   **当天自动拦截 2 次。**
2. **并发跑两个 `ab_restart.sh`** → 互相 kill,服务空窗约 2 分钟。同上,已拦截。
3. **在运行中的代码目录 `git checkout`** → 服务崩(`ModuleNotFoundError: moe_align`,
   懒加载,启动正常、首个请求才炸)。
4. **诊断代码引用了别的类的属性**(`self._swa_layers_num` 不在 `DSV4PoolConfigurator` 上)
   → 服务起不来。**教训:改生产代码前先确认属性归属,已用脚本核对 class 归属。**
5. **OOM 归因到错的分配上** —— 以为 `k16 = k_fp8.to(fp16)` 是主因,改完崩溃点不变。

**方法论教训:写文档约束不了我自己,必须落到机制。** 第 1、2 类写成文档后当天仍犯,
加进 `ab_restart.sh` 才真的拦住。

## 九、★ 结论:256K 与 F3 已同时上线

三道闸门全过(配置:`mem_fraction_static 0.97` + `SGLANG_PP_LAYER_PARTITION=11,11,11,10`
+ `chunked_prefill_size 256` + SWA 计价修复 + `_CHUNK 256`):

| 闸门 | 结果 |
|---|---|
| greedy parity | **6/6 逐字节一致**(vs SWA 修复前的 F3 臂) |
| 256K prefill | **258,941 token 跑通**,413 s |
| 200 题 GSM8K 并发 8 | **acc 0.945**,`unhandled cuda error` / `out of memory` / IMA / `Scheduler hit an exception` **全 0**,服务存活 |
| 纯默认启动复核 | 进程 env 含 `SGLANG_PP_LAYER_PARTITION=11,11,11,10`,池 267,776,parity 6/6,烟测正常 |

| | 之前 | 之后 |
|---|---|---|
| KV 池 | 162,304 | **267,776**(+65%) |
| 可用上下文 | 162K(配置写 256K) | **≥256K** |
| prefill | 基准 | **+7.6%**(F3 完整保留) |

**做法**:SWA 计价只算本 stage 真实的 SWA 层(PP0 从 1944.75 → 1109.25 字节/token),
腾出的显存变成更大的池;池变大又吃掉 PP0 余量(0.64 → 0.46 GB),而 260K prefill 需要余量,
于是把两个 prefill 瞬时量减半(`chunked-prefill-size` 512→256 让索引器
`[Q, max_seqlen_k]` fp32 从 133 → 66 MiB;`_CHUNK` 1024→256 让 score tile 从 64 → 16 MiB),
合计 ~115 MiB,而崩溃时缺 57 MiB。两项都是纯分块,数值逐位不变。

**注意因果**:余量是池之后的剩余 —— 池变大必然吃掉余量。所以"降价率"和"抬 mem-fraction"
都会让 PP0 更紧张,必须一起看。

## 十、★ 上线前抓到的第二个"配置不生效"bug

`SGLANG_PP_LAYER_PARTITION="${SGLANG_PP_LAYER_PARTITION-11,11,11,10}"` 写在 launcher 里,
但**没有 `export`**。sglang 是用 `os.getenv` 在 `get_pp_indices` 里读它的,
**不是 server arg** —— 未 export 的 shell 变量根本到不了服务进程。

F3 之前一直生效,是因为每次都由命令行传入(那是真正的环境变量)。
**若直接跑 `deepseek-v4-flash.sh`,会得到没有 F3 的版本:prefill 少 7.6%、KV 池更小。**
已加 `export`,并在注释里写明原因。

**这是今晚第二个"看起来对、实际不生效"的 bug**(第一个是 SWA 计价高估)。
**教训:默认值和配置项必须验证"它真的被消费了",不能只看它写对了没有。**
纯默认启动(不带任何 env)复核,是唯一可靠的检验。

## 十一、明天的第一优先

**`_topk_transform_paged_triton_kernel` 的长上下文开销**(§二):150K 上下文下涨 34.5 倍,
是 PP3 上最大的非通信 kernel(2.15 ms/token);而 gemm/moe 完全不随长度变。
这是"用了 256K 之后 decode 会不会更慢"的关键,与本文的配置问题完全独立。

