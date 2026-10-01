# 2026-10-01 — 调度形态复审:多条建议被证否,一条新杠杆测出是负的,一处生产故障定到内核

工作区 `plan2026-09-30/`,测量对象 = 生产 server(`:8200`,TP2×PP4,`MAXREQ=8`),
launcher = `/data/nvme/sglang/deepseek-v4-flash.sh`,sglang 是 **editable 安装直接指向
`sglang-codex/sglang/python`**,所以改源码 + 重启即可 A/B,不需要拷文件。

---

## 0. 本轮结论速览

| 编号 | 原建议 | 结论 | 证据 |
|---|---|---|---|
| **D1** | 抬 `--cuda-graph-max-bs-decode` 到 4/8 | **证否(原形式)** | decode 批永远 ≤2,graph 覆盖已 100% |
| **§4** | 缩 `--max-total-tokens` 一次性解锁三件事 | **证否** | KV 池总预算只有 **0.31 GB**,flag 根本没绑定 |
| **D5** | `num_continuous_decode_steps` 摊薄调度 | **证否** | 只在 `arg_groups` 定义,**全仓无消费点**,是死 flag |
| **chunk 512→1024** | 加大 chunk 摊薄 90 ms 气泡 | **证否(已实测)** | `PREFILL_PROFILE.md:657`:1419/1410/1411,与 512 持平 |
| **D1'** | 抬 `MAXREQ` **与** graph 桶**同时** | **⚠ 部分成立:+5.8% decode(仅 N≥16),N=8 反而 −4.9%;模型 +52% 被证否** | 见 §3;**已关闭**,不再往 32 走 |
| **新杠杆** | `SGLANG_PP_EARLY_PROXY_SEND=1` | **❌ 性能 −27~30%,保持 0** | 三臂交错 A→B→A′,bs=1 对照 0.992,见 §4 |
| **新故障** | 生产默认配置下 CUDA IMA 崩溃 | **✅ 已修复**(根因 `topk.py` 缺 `page_id < page_table_width` 上界):`topk.py` 的 `_topk_transform_paged_triton_kernel` 里 **`page_table_width` 传入但从未使用** → 脏 page id 经 `extra_indices` 流进稀疏 attention(后者又缺 `num_pages` 上界)→ IMA;**SV2 夹紧后 IMA 消失、greedy 6/6** | §6.5→§6.7;取证靠 **launch 后 device-wide sync** 拿到的两份可信栈;连证否:懒加载 / headshared 单独首因 / decode graph / prefill graph / coredump 工具链 |
| **F3 层重排** | `SGLANG_PP_LAYER_PARTITION=11,11,11,10` | **✅ 实测 prefill +4.1~+10.0%(均值 +7.6%),建议上线** | 纯配置零代码;瓶颈 stage 从 PP3(274.6 ms/chunk)挪走;8/8 格子全正、bs=1 对照 0.999、跨臂字节一致、200 题 GSM8K 零崩溃。**旧文档的"不可用"结论已过期**(当时 PP0 余量 0.42 GB,现 0.64 GB)。见 §6.9 |
| **EAGLE MTP** | 投机解码 = 最大单项 | **❌ 算术上不可能** | draft 权重 **5.34 GiB/卡** 复制到每卡;总权重 164.91 GB / 容量 176 GB = 93.7%。旧文档归因("1010 MiB embedding")**是错的**。见 §6.10 |
| **prefill graph** | prefill CUDA graph = prefill 主杠杆 | **❌ 证否:慢 2.9 倍** | `breakable` 实测 **0.368×**(8/8 格子 0.34–0.36,noise 2.8–6.7%;decode 持平、bs=1 对照 0.993);`full` **dsv4 未实现**、启动即崩;`tc_piecewise` 未测。见 §6.8 |

### 四个可执行结论

1. **上线 F3 层重排**(`SGLANG_PP_LAYER_PARTITION=11,11,11,10`):**prefill +7.6%**,
   零代码、全部闸门通过(§6.9)。这是本轮唯一的性能收益。
2. **`SGLANG_PP_EARLY_PROXY_SEND` 保持默认 0** —— 它出现在 launcher 注释、`ab_early_proxy.sh`、
   opt4 commit 里,一直像个待开的开关;隔离 A/B 证明它在多请求 decode 上是 −30%。
   当年 "25.7 vs 26.1" 之所以没发现,是因为**只测了 bs=1,而 bs=1 正是它唯一无影响的点**。
3. **prefill graph 与 EAGLE 两条大杠杆都已关掉,不要再投**:
   - **prefill CUDA graph 慢 2.9 倍**(§6.8),`full` 后端 dsv4 未实现;
   - **EAGLE 算术上不可能**(§6.10):draft 权重 5.34 GiB/卡,总权重占容量 93.7%。
   - 剩下唯一还有较大理论空间的是 **attention 的 440 ms(占 prefill 50.5%)**,
     但它要绕开 Triton 手写 CUDA/PTX,不是调参能解决的。
4. **那处出厂 IMA 已修复**(§6.7):真因是 `topk.py` 缺 `page_id < page_table_width` 上界。
   取证过程本身是本轮的第二个成果 —— **异步 CUDA 错误的 Python 归因只有在同步链路里可信**
   (`CUDA_LAUNCH_BLOCKING` 不覆盖 driver API;`SGLANG_TRITON_SYNC_EVERY_LAUNCH` 才对)。
   当时的中间结论("Python 栈不可信"、"改用 coredump")现已被推翻,保留在 §6.5–§6.7 作教训:
   同一个 IMA 在四份档案里给出**四个互不相干**的报告帧 —— `dequant_block_fp8_slice`
   (A2,纯生产)、`dequant_block_fp8_slice`(PG4,headshared 关)、
   `_headshared_sparse_decode`(LBO1)、`moe_swiglu_clamp`(LBO2)。
   **四个全是 eager Triton launch** → 那是"粘性错误落到下一个 launch"的签名,不是首因。
   **`CUDA_LAUNCH_BLOCKING` 只改变报出时机,不改变可信度**:PG3 与 LBO1 同配置,
   唯一差别是 blocking,一个连 Python 栈都没有。已证否:服务期懒加载、headshared、
   decode graph replay、prefill graph(§6.5→§6.7)。
   → **第四条信源也堵死:driver 级 CUDA coredump**。`SGLANG_CUDA_COREDUMP=1` 成功落盘
   2×199,755,747 B(写完整、符号与行表俱全),但 **`e_phnum = 0` → cuda-gdb
   12.0 / 12.9 / 13.2 + 系统 gdb 全部拒绝**;最小越界复现证明**与 sglang 的
   `skip_*` flags 无关**,`log_only` 环境变量也**不生效**(§6.7)。
   → **第五条(现行):SY1 臂** —— 在 `CudaLauncher.__call__` 之后**立即 device-wide sync**,
   让粘性错误**最多晚一个 launch** 就抛出,拿可信 Python 栈(§6.7)。
   ⚠ 四条方法论教训:**(a) 同秒相关 ≠ 因果;(b) 换配置后栈会换,单次栈不能当首因;
   (c) 异步 CUDA 错误的 Python 归因只有在同步链路里可信,而 `CUDA_LAUNCH_BLOCKING`
   只管 CUDA Runtime、Triton 走 driver API(`cuLaunchKernel`);
   (d) 现成工具在"CUDA 13.2 + 零 program header"组合下会静默不可用 ——
   先用最小复现验证工具本身,再拿它查真问题。**

---

## 1. 证否 D1:decode 批永远 ≤2,graph 已 100% 覆盖

`scheduler.py:1194-1199`:

```python
if not get_parallel().pp_max_micro_batch_size:
    pp_max_micro_batch_size = max(self.max_running_requests // pp_size, 1)
```

`MAXREQ=8 × PP=4 → pp_max_micro_batch_size = 2`,服务器自报已确认
(`/get_server_info`:`"pp_max_micro_batch_size":2`,`"cuda_graph_max_bs_decode":2`)。

**4 个 PP stage 的日志直方图完全一致**(整份日志,1336 行 Decode):

```
#running-req : {1: 704, 2: 632}        ← 永不超过 2
#queue-req   : {0: 1336}                ← 恒 0
cuda graph   : {'True': 1336}           ← 100% 覆盖
#new-seq     : {1: 124, 2: 112}         ← prefill 同样被 ≤2 卡住
```

`metrics_reporter.py:907` `num_running_reqs = len(batch.reqs)` —— 这个字段是 running batch
的真实长度,不是采样出来的。抬 graph 桶到 4/8 只会捕获**永远不会被 replay** 的图。

### 一个必须澄清的反直觉点

8 个并发请求的 latency 是 **14.95 / 14.97 / 14.98 s**(极差 0.03 s),聚合 136.7 tok/s ——
**真并发,没有排队**,但 `#running-req` 却恒 ≤2。

两者不矛盾:`#running-req` 报的是**微批大小**,8 个请求被切成 4 个微批在 4 个 stage 上流水。
`get_num_allocatable_reqs()`(4 个调用点全在 prefill 路径,`adder.can_run_list`)管的是**准入**,
已入 running batch 的请求不会被摘掉。

---

## 2. 证否 §4:KV 池根本没有 270000 token 可缩

启动日志的 DSV4 内存计算(每 stage 相同的 `available_bytes`):

```
bytes_per_full_token=1457.00 / 1944.75, available_bytes=0.31 GB, full_token=153600
Load weight end  avail=0.94 GB   (PP3)   ← 权重 20.15 GB / 共 21.09 GB
Memory pool end  avail=0.93 GB   (PP3)   ← 池只吃了 0.01 GB
```

- `--max-total-tokens 270000` **不生效**:实际 `max_total_num_tokens=153600`,由显存容量决定。
- 池的**全部预算 `available_bytes=0.31 GB`**,分摊到每 stage ≈0.077 GB。
- 权重 18.44–20.15 GB,才是显存的主体。

**→ 缩池最多挤出 0.3 GB,而且不是 PP3 那 0.79 GB 的主要来源。这个旋钮作废。**
显存侧的正解回到 **F3 层重排**(PP3 有 0.75 GB 未对账)。

(另注:`/data/nvme/sglang/logs/serve-prod.log` 与 `/data/nvme/sglang-codex/logs/serve-prod.log`
是**两个不同文件**,mtime 分别是当天与 9/29,`max_total_num_tokens` 分别是 153600 与 269824 ——
之前混用过,读数以 `/data/nvme/sglang/logs/` 为准。)

---

## 3. decode 吞吐模型(4 点验证通过)→ D1'

```
aggregate = min(M, P) × bs / (P × stage_time(bs))
P = 4 (PP),  bs = 微批大小,  M = ceil(N / bs) = 在飞微批数
```

| N(并发) | 模型预测 | 实测(dec_probe 中位数) |
|---|---|---|
| 1 | 26.7 | **26.7** |
| 2 | 34.2 | **35.5** |
| 4 | 68.5 | **66.8** |
| 8 | 137 | **105.9** / **136.7**(热态/冷态) |

反解出 `stage_time`:

```
stage_time(bs=1) = 9.4 ms      a ≈ 4.1 ms(权重读 + 固定开销)
stage_time(bs=2) = 14.6 ms     b ≈ 5.2 ms/请求(KV/attention + MoE 专家读)
```

**每请求流水延迟 = P × stage_time = 4 × 14.6 = 58.4 ms/token**,
与实测 `14.97 s / 256 tok = 58.5 ms/token` **完全吻合**。

### 由此得到的三个硬结论

1. **N=8 已经是饱和点**(`M = N/bs = 4 = P`)。再抬 `max_running_requests` 在 bs=2 下**无收益**。
2. **bs=2 在 N=8 时是最优**:

   | bs | 需要的 N(保持 M≥P) | aggregate |
   |---|---|---|
   | 1 | ≥4 | 106.8 |
   | **2** | **≥8** | **137** ← 当前 |
   | 4 | ≥16 | 8/(4×stage_time(4)) |
   | 8 | ≥32 | 8/(4×stage_time(8)) |

   **所以"只抬 graph 桶"不但无效,单独抬 `pp_max_micro_batch_size` 在 N=8 下还会变慢**
   (bs=4、N=8 → M=2 < P=4 → 流水半空 → 模型预测 79.7 tok/s,比 137 还差)。
3. **唯一的 decode 杠杆是 `stage_time`** —— 除非 N 和 bs 一起抬。

### D1'(待验证):N 与 bs 必须同时抬

`stage_time` 从 bs1→bs2 只涨 56%(工作量翻倍)→ 次线性。若趋势延续:

```
MAXREQ=32 → pp_max_micro_batch_size = 32//4 = 8
GRAPH_BS="1 2 4 8"  GRAPH_MAX_BS=8
N=32, bs=8, M=4=P  →  aggregate = 8 / stage_time(8)
线性外推 stage_time(8) = 4.1 + 5.2×8 = 46 ms  →  174 tok/s  (+27%)
```

**这是一个有明确正反预测的实验**:若 `stage_time(8) > 64 ms` 则证伪。

### D1' 结果:实测 +5.8%,但模型的 **+52% 被证否** → 已关闭

臂 `M16`:`MAXREQ=16` + `GRAPH_BS="1 2 4"` + `GRAPH_MAX_BS=4`(故
`pp_max_micro_batch_size = 16//4 = 4`),`DEC_BS=1,2,4,8,16`、`./measure.sh M16`,
冷启 + 与 A 臂完全同脚本。第一次尝试撞 IMA 崩掉(`res/crash_M16_ima.log`),
第二次(`res/measure_M16b.log`)贪心 parity 6/6 逐字节一致。

| N | 基线 A(MAXREQ=8) | M16 | 比值 |
|---|---|---|---|
| **1(内部对照)** | 26.5 | 26.7 | **1.008 → 两臂热可比,下表可信** |
| 2 | 41.3 | 42.7 | 1.034 |
| 4 | 67.6(spr 7.1%) | 73.4(spr **28%**) | 1.086 NOISY |
| 8 | **105.6** | 100.4(100.0/100.4/103.8) | **0.951 → 慢 4.9%** |
| 16 | 不可测(N>8 被 MAXREQ 卡住) | **111.7**(107.5/111.7/114.8) | 较基线最优点 **+5.8%** |

三点判读:

1. **N=8 是纯对照,结论是负的**。两臂在 N=8 都能凑满 M=4 个在飞微批,M16 只是把每批装得更大,
   结果 −4.9%。→ **"在现有并发下抬 MAXREQ 有收益"不成立。**
2. **N=16 的 111.7 vs 模型预测 161 = −31%,正预测被证否。**
3. 反解服务时间 `st = min(M,P)×bs/(P×agg)`(M×bs≈N 时该式对 bs 的假设不敏感):
   N=1→9.4 ms,N=2→11.7,N=4→13.6,N=8→19.9,N=16→35.8 ms。
   折算**每请求**:9.4 / 9.4 / 9.4 / 9.9 / **8.95** ms。

**模型错在哪**:文档给的 `stage_time ≈ 4.1 + 5.2×bs` 是次线性的(→ bs=4 应为 24.9 ms),
实测却近似**线性**(bs=4 的 35.8 ms 里每请求仍要 8.95 ms)。代回原式:

```
agg = min(M,P) × bs / (P × stage_time)  ≈  min(M,P) / (P × 9.4 ms)
```

→ **吞吐只取决于在飞微批数 M 是否 ≥ P,与 bs 几乎无关**。M≥4 时天花板 ≈ 106–112 tok/s ——
这正好解释了基线 N=8(105.6)和 M16 N=16(111.7)为什么都落在同一处。

**两条可直接用的推论**:

- **`MAXREQ=32` / graph `bs=8` 那一臂不必跑**:按每请求 8.95 ms 再降 5% 外推,顶多 ~117(+4%),
  却要再烧一档 KV 图 + 要求客户端真有 32 并发。**D1' 就此关闭。**
- **decode 不是权重读带宽受限,而是每请求受限**(KV 读与命中专家读随请求数线性增长)。
  所以**调度类杠杆到此为止,decode 剩下的唯一杠杆是削减每请求的 kernel 工作** ——
  即 §9-C.3 的 MoE k-split。这反过来印证了文档把 k-split 排在 decode 首位是对的。

### 代价与混杂

- `max_total_num_tokens` **153600 → 142848(−7%)**:捕获 bs=4 的图烧 0.18 GB
  (基线 bs=2 为 0.13 GB),PP3 余量仍 0.74 GB,**未触发新的 OOM**。
- prefill 八格一致 **−2.4~2.9%**(如 4000/K=8:1420.8 → 1380.0)。
  **但这条有混杂**:M16 臂 decode 探针多了 N=16 一档,总 token 量比 A 臂多 25%,
  载荷历史更重,而本机载荷历史值 ~25%(§3)。**无法判定是配置造成**,
  要判必须让两臂 prefill 探针之前的历史一致 —— 记为 *存疑*,不作为决策依据。

### 补:这些数字的结构来源(`#running-req` 报的是微批,不是整批)

`report_decode_stats(running_batch=batch)` 的 `batch` 是**当前微批**;PP 事件循环
(`scheduler_pp_mixin.py:115-141`)每轮换入一个槽:

```python
self.pp_loop_size  = pp_size + pp_async_batch_depth   # 4 + 2 = 6 个微批槽
self.running_mbs   = [ScheduleBatch(reqs=[]) for _ in range(self.pp_loop_size)]
for mb_id in range(self.pp_loop_size):
    self.running_batch = self.running_mbs[mb_id]      # ← 换入本槽的批
    ...
```

所以三条互相矛盾的观测在这里统一了:

- `#running-req ∈ {1,2}`(不是 8)—— 它是**微批**大小,上限 `pp_max_micro_batch_size = MAXREQ//PP = 2`;
- `#queue-req` 恒 0 —— 另外 6 条请求不排队,它们在**其余 5 个槽**里在飞;
- `cuda graph: True` 100% —— 走 graph 的就是那个 ≤2 的微批,所以"batch>2 会走 eager"
  这句文档断言**在本配置下从未发生过**(它把 `#running-req` 当成了整批)。

**由此,文档那句 "Raising the graphed batch range is untested and is the cheapest thing left"
真正的变量不是 graph 桶(桶无效,§1),而是 `MAXREQ` —— 它才决定 `pp_max_micro_batch_size`,
即每个 forward 装多少请求。这就是 D1' 要动的量。**

### 模型与新测数据的偏差(诚实记录)

同一脚本 `measure.sh A`、3 轮中位数:**26.5 / 41.3 / 67.6 / 105.6**(模型 26.7/34.2/68.5/137)。

- N=1、N=2、N=4 吻合(±15%);
- **N=8:105.6 vs 模型 137,低 23%**。反解 `stage_time` 随 N 变大:
  `N=2 → 12.1 ms`,`N=4 → 14.8 ms`,`N=8 → 18.9 ms`。

**即"每微批的服务时间"随在飞微批数增长**,模型里没有这一项。两种可能:
(a) host CPU 竞争(8 个 scheduler 进程,`disable_overlap_schedule=True` 下 CPU 与 GPU 串行;
本机实测各占 71–80% 单核),(b) 我的 `min(M,P)` 闭合排队近似不够。
**这正是 D1' 要测的东西**:抬 `MAXREQ` 会同时抬 M,若 (a) 成立,D1' 反而会更差。

> 另注:文档自己的并发表(1/2/4/8 → 25.2/39.5/65.6/**132.2**)在 N=8 上比我高 26%,
> 而 N≤4 基本一致。差距来源未定(热态/冷态:本机冷启 `pp_probe` 曾测到 **136.7**,
> 与文档的 132.2 吻合,而同一实例跑完探针后掉到 105.6)。
> **载荷历史在本机值 ~25%,远大于文档假设的 6–10%** —— 所以每臂必须冷启 + 同脚本。

---

## 4. 新杠杆:`SGLANG_PP_EARLY_PROXY_SEND`

`scheduler_pp_mixin.py:179-220` 两条路径:

- **默认(0)**:`_pp_send_dict_to_next_stage` 在 `_pp_process_batch_result` **之后**发
  → 每个 stage 的 CPU 尾巴(阻塞 recv + `d2h_event.synchronize()` + 结果处理)都压在下游关键路径上。
- **=1**:forward launch 后立刻异步发出(`ready_event=self.launch_event`),
  把这段尾巴与计算重叠。

它是 **opt4 的一部分**(commit `a622747b92`:"scheduler_pp_mixin: SGLANG_PP_EARLY_PROXY_SEND
starts the proxy send before the result post-processing"),但 `environ.py` 默认 `False`,
生产脚本显式 `export ... :-0` → **从未被单独 A/B**(现有 `ab_opt4_prefill.sh` 只比 elementwise.py)。

仓库里已经有为此写的 `ab_early_proxy.sh`,而且写得很严谨(绝对路径 bug、flag 校验、
"旧 server 还在监听就跳过"的防呆),**但 `PROGRESS.md` 与全部 md 里 grep 不到任何结果记录**。

风险:commit message 明确讨论过 fencing 危险并已处理(`send_proxy_requires_forward_fence`),
但仍需 **greedy parity 校验**(`plan2026-09-30/greedy.py`)。

### 结果:正确性通过,但性能是 **−27~30% 的回归** → 保持 0

三臂交错(A → B → A′),每臂都经 `measure.sh <TAG>` 全套同历史(冷启 → greedy×2 → dec → pf),
用 `ab_compare.py A B` 比对。decode 聚合吞吐,3 轮中位数:

| bs | A (flag=0) | B (flag=1) | B/A | 组内 spread |
|---:|---:|---:|---:|---:|
| 1 | 26.5 | 26.3 | **0.992** | 2.4% / 1.5% |
| 2 | 41.3 | 30.2 | **0.732** | 14.5% / 6.9% |
| 4 | 67.6 | 48.1 | **0.711** | 7.1% / 4.3% |
| 8 | 105.6 | 74.1 | **0.702** | 4.3% / 29.2% |

prefill 侧同样全线略负(比值 0.966–0.982,即 −2~3%)。

**为什么这不是热漂移**(这是本机 6–10% 漂移的典型陷阱):

1. **内部对照 bs=1 = 0.992** —— 两臂可比;
2. **同臂内 round3 > round1**(+2.4% ~ +29.9%),后期**更快**,趋势与"发热衰减"相反;
3. **B 后跑**,按上述"越跑越快"B 应占便宜,结果反而最慢 → 回归真实且**很可能被低估**;
4. **bs≥2 三个比值 0.732 / 0.711 / 0.702 高度一致** —— 系统性效应,不是单点噪声。

**为什么当年没发现**:`SM75_DSV4_DECODE_PROFILE.md:60-64` 记的 "25.7 vs 26.1 tok/s"
**恰好只测了 bs=1** —— 而 bs=1 正是这个 flag 唯一无影响的点(比值 0.992)。

**机制**(与结构吻合):两条路径调用**同一个** `_pp_send_dict_to_next_stage`、参数完全相同,
`send_proxy_requires_forward_fence` 也一样 —— 唯一差别是**执行顺序**。early 把 NCCL send
提前到 `d2h_event.synchronize()` **之前**,即让 comm 流与 forward 的计算流**并发**。
`pp_async_batch_depth=2` 下 4 个微批连续在飞 → 重叠窗口常开 → NCCL kernel 与计算 kernel
抢 SM;bs=1 只有 1 个微批、有大量空档,所以无代价。**这正好解释"bs=1 不动、bs≥2 全线 −30%"。**

> **动作:`SGLANG_PP_EARLY_PROXY_SEND` 保持默认 0。** 这个 flag 出现在 launcher 注释、
> `ab_early_proxy.sh`、opt4 commit 里,一直像个待开的开关 —— 现在有隔离 A/B 证明它该关。

---

## 5. prefill 被 `chunked-prefill-size=512` 串行化

并发预填充测试(3 轮中位数,`pf_conc_probe.py`):

| prompt | K=1 | K=2 | K=4 | K=8 | 线性理想 |
|---|---|---|---|---|---|
| ~1454 tok | 793 | 1019 | 1199 | **1312** | 11632 |
| ~5879 tok | 1197 | 1315 | 1385 | **1416** | 9574 |

窗口内调度行为:

```
#new-seq   {1: 350, 2: 16}        ← 366 次里只有 16 次批到 2 条
#new-token {512: 335, 503: 31}    ← 恒 512 = chunk 上限
#queue-req {0: 66, ..., 7: 24}    ← 排到 7 条
#pending   {… 45496}              ← 待处理 token 堆到 4.5 万
```

**因为 prompt(≥430 tok)单条就超过 512 token 预算,一个 forward 只能装 1 条序列** →
prefill 逐 chunk 串行,并发只买到 **+18%**。

天花板 ≈ **1400 tok/s**,与 `PREFILL_PROFILE.md` 的单流 8000/13000 → 1080/1361 一致:
**长 prompt 已经贴着天花板**,短 prompt(793)和高并发才吃亏。

### 更正:chunk=512 不是 OOM,是"零收益"

我一度把 `chunk 512→1024` 归因为 warmup OOM(进而推论"需先有显存余量"),**这是错的**。
`PREFILL_PROFILE.md:657` 的实测是:

> chunk=1024(pool 230k)**能跑**,但 tok/s 与 512 相同甚至略低(**1419 / 1410 / 1411**)。
> 我之前推理"每 chunk 90 ms 气泡,chunk 加倍能摊薄省 12%"是**错的** —— 那 90 ms 是
> **pipeline skew,正比于每 chunk 的计算量,不是固定开销**。流水线已经 95% 满载,
> 加大 chunk 只会让气泡等比例变大。

(另外 launcher 第 64–67 行那条"CHUNK 不得 >512"的注释,理由是 `_merge_partial_attn` 的
`[tokens,128,512]` fp32 临时量在 2048 tok 时每层 >1 GiB 会 OOM —— 而消除该临时量的
`SGLANG_SM75_FUSE_ATTN_TAIL`(commit `a93bd25001`,9-30)**默认已是 True**,生产在用。
所以那条注释的机制也已过时;但既然 chunk 加大本身零收益,两条都不构成方向。)

**→ prefill 结论:在 PP4 + chunk=512 这个拓扑下,长 prompt prefill 已经 compute-bound、
流水线 95% 满载,继续抬 chunk / 抬并发都没有空间。唯一剩余的 prefill 杠杆是 kernel 本身
(opt1–opt4 那条线,已 +6.4/7.7%)。**

---

## 5.1 更正:all-reduce「第 4 项」的机制是错的,它不是 open 项

两份文档冲突,后一份更细更晚:

| | `SM75_DSV4_DECODE_PROFILE.md:151`(较早) | `PROGRESS.md:350` #4(9-30,prof2b 实测) |
|---|---|---|
| 说法 | EXTEND 一次 4 MB 事件 0.216 ms = 18.5 GB/s,只到 NVLink 87 GB/s 的 1/5 → 提速 kernel 省 3.7 ms/chunk(~6%) | 逐事件对齐两 rank 的 AR 启动时刻:偏移 **+2.4~+9.5 ms ≈ 领先侧自旋时长**;**落后侧 AR 真实耗时中位 25 µs**(peer 早已把数据推进本地 HBM) |
| 瓶颈 | kernel 效率 | **rank 间 skew**;wire 从不是瓶颈(≤16.8 MB 在 NV2 理论 ~0.6 ms) |
| 形态 | "the only interconnect-adjacent item left" | 已是 Lamport one-shot push,**单份拷贝**已是现状 |

**判读:0.216 ms 里绝大部分是等 peer,不是把 4 MB 搬完的时间。**
把传输提到链路速度不会缩短"等对方到达"的部分 → **那个 +6% 是把"事件总时长"
当成了"传输时长"** —— 与 §2 的 `--max-total-tokens`、§5 的 chunk 属于同一类错误:
**测错了旋钮**。我上一轮把它列进优先级表,已更正(§9-C.2)。

`PROGRESS.md:393` 自己给的候选才是 prefill 剩下的方向:

| 候选(PROGRESS #4) | 状态 |
|---|---|
| **prefill CUDA graph 分桶** | **本轮正在测可行性**(被 MLA 1536 MB 预留挡,见 §6.3) |
| 降 launch 数 | 与上一条同源 —— graph 就是把 launch 摊掉 |
| 自旋 kernel 小 grid 化 | 改 `custom_all_reduce.cuh`(**C++,在 wheel 里**),不在 Python 层 |
| 慢 rank CPU 采样 | 纯诊断 |

---

## 6. 新发现:生产默认配置下一次 CUDA IMA 崩溃(文档无记录)

**现场已存档 `res/crash_A_ima.log`。** 时间线:

```
00:29:31  实例启动(A2 重启, SGLANG_PP_EARLY_PROXY_SEND=0 = 生产默认)
00:30:19  捕获完成, PP3 avail 0.79 GB
00:30:24  第一个 /generate + /freeze_gc
00:30:32  00:30:35 00:30:37 00:30:40 00:30:42  其余 /generate 全 200 OK
00:30:42  PP3: Triton kernel '_expand_prefill_causally_kernel' device-loaded
          (free device mem: 0.65 GiB)
00:30:42.543  [rank4] [PG ID 6 GUID 59(pp:device) Rank 2]
              CUDA error: an illegal memory access was encountered
00:30:42.544  [rank5] [PG ID 6 GUID 61(pp:device) Rank 2]  同上
00:30:43  PP1 TP0/TP1 抛 gloo "Connection closed by peer"  ← 级联受害者
00:30:43  SIGQUIT → 进程树退出
```

- 实例只活了 **72 秒**,只处理 **24 个 prefill batch**(=6 prompt × 4 stage)。
- 根因是 **CUDA IMA**,不是 gloo/gloo;gloo 报错是下游。
- **`grep -ril "illegal memory"` 在全部 md 与全部历史日志里只命中这一个文件** → 全新故障。

### 已排除的

| 嫌疑 | 结论 |
|---|---|
| opt4 改了 PP 共享路径 | **排除**。`git show a622747b92` 对 `scheduler_pp_mixin.py` 的改动在 flag=0 时化简为原逻辑(`if not early_proxy_send and not is_last_rank` → `if not is_last_rank`),行为完全等价 |
| 两条路径 fence 不对称 | **排除**。`_pp_send_proxy_to_next_stage` 与 early 分支**都**设 `send_proxy_requires_forward_fence = result.can_run_cuda_graph` |
| Triton 懒加载 | **不充分**。懒加载告警是慢性的:`p2-t8p1` 88 次、`final150` 80 次、`rx-norx` 66 次、`serve-final` 48 次…… 这些运行都没崩 |

### 仍然成立的线索

1. IMA 与 `_expand_prefill_causally_kernel` 的**首次 device-load 同毫秒发生**。
2. 项目自带的 `srt/utils/triton_load_watch.py` 文档写明失效类:triton 在
   `CompiledKernel._init_handles -> cuModuleLoadData` 才把 cubin 上卡,
   **这笔内存在 torch caching allocator 之外**;引擎把池子撑到 init 后没余量,
   "a specialization first used mid-serving ... can die ... minutes or hours in"。
   本次 `free device mem: 0.65 GiB` < 它自己的 1 GiB 告警阈值。
3. 本次 40 次懒加载全部发生在 **0.65–0.79 GiB** 区间(PP3 最紧),覆盖 16 个不同 kernel。
4. 有开关 `SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY=1` 可把每次晚期加载变成硬失败,
   用于判定"预热是否覆盖全部特化"。

**判读**:懒加载足以解释"时间点巧合",但不足以单独致崩(否则历史早该崩)。
更可能是**懒加载的 >1 s 编译占住 GIL / 打乱时序,暴露 PP proxy 的潜在 ordering 问题**。
在拿到第二次复现之前,**不下结论**。

### 第二次复现(01:06:33)—— 拿到了,签名完全一致

现场存档 `res/crash_A2_ima.log`(launcher 用 `>"$log"` 截断,崩溃后必须先 `cp` 再重启)。
**这次抓到了带完整 Python 栈的首因**:

```
deepseek_v2.py:363  gate_up, _ = self.gate_up_proj(x)      ← shared experts
  → fp8.py:1259     w8a16_linear(x, layer.weight, layer.weight_scale_inv, bias)
  → fp8_w8a16.py:617  dequant_block_fp8_slice(
  → fp8_w8a16.py:424  _dequant_block_fp8_kernel[(cdiv(n,64), cdiv(k,128))](
triton/backends/nvidia/driver.py:328  self.launch(...)
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

两崩溃对照:

| | 懒加载 | IMA | 间隔 | IMA 位置 / PG | 有无 Python 栈 |
|---|---|---|---|---|---|
| #1 `crash_A_ima.log` | 00:30:23 `×4` | 00:30:42.543 | **19 s** | rank4/rank5,`PG 6 GUID 59/61 (pp:device) Rank 2` | 无 |
| #2 `crash_A2_ima.log` | 01:06:20 `×4` | 01:06:33.910 | **13 s** | rank5,`PG 2 GUID 7 (tp:device)` | **有 → `_dequant_block_fp8_kernel`** |
| #3 `crash_M16_ima.log` | 01:21:16–17 | 01:21:32.000 | **15 s** | rank4/rank5,**与 #1 同 PG** | 无(NCCL watchdog 先于主程) |

**#1 与 #3 是同一对 rank、同一个 process group** —— 复现到这个程度已经不像巧合。

⚠️ **一个必须诚实记录的不一致**:懒加载告警**全部打在 PP3**,而 #1/#3 的 IMA 在
rank4/rank5。原因是 `_on_kernel_load` **只在 `free_gb < 1.0 GiB` 时才告警**
(`triton_load_watch.py:108`),而本机只有 PP3 的余量(0.74 GB)低于阈值 ——
**其余 stage 同样在懒加载,只是不打日志**。所以"告警在 PP3"不等于"只有 PP3 加载",
这条线索不能用来定位故障 stage。

**5 个实例崩了 3 个**(实例 1、4、5),都发生在**冷启后的第一批请求**上:
`greedy_A21/A22`、`greedy_M161/M162` 全部 `Connection refused`。

两次都是 **PP3 的 TP0+TP1**,同一对称、同一 `free device mem` 档位。

**#2 里 01:06:20 是一次集中的 38 个内核懒加载**(同一秒、全部 PP3),
`free device mem` 从 0.70 → 0.68 → 0.66 GiB 一路被吃:

```
compute_position_kernel / _expand_prefill_causally_kernel / _page_table_positions_kernel
_init_compressed_attn_metadata_kernel / _dequant_block_fp8_kernel(×2)
_mqa_paged_smallq_kernel / _topk_transform_paged_triton_kernel
_headshared_sparse_kernel(×4) / _wo_a_absorb_kernel / _moe_align_small_numel_kernel
alloc_extend_kernel / alloc_decode_kernel / get_and_clear_swa_pages_kernel ...
```

### 为什么这些内核必然"服务中途才首次使用"

**看启动器:`--cuda-graph-backend-prefill disabled`。** decode graph 在启动期 capture,
把 decode 路径的内核全部编译+上卡;**prefill 不做 capture,于是 prefill/extend 路径的
一个内核都没加载过** → 必然等到第一个请求才懒加载。上面那 38 个名字里
`_expand_prefill_causally_kernel`、`alloc_extend_kernel`、`_init_compressed_attn_metadata_kernel`
全是 prefill 路径;`_dequant_block_fp8_kernel` 更是**只在 prefill 分支跑**
(`_w8a16_linear_impl` 的 `else`,注释写明 "Prefill: dequantize a tile … and let cuBLAS have it",
按行分块循环 + 每次临时 `torch.empty` scratch)。**两次崩溃都在 prefill 负载下,与代码吻合。**

**为什么会崩**:`triton_load_watch.py` docstring 自己写明 —— cubin 上卡走
`cuModuleLoadData`,**这笔内存在 torch caching allocator 之外**;PP3 权重后只剩
0.79 GB、批量加载后 0.66 GB,**已低于它自己的 1 GiB 告警阈值**。

### 仍然存疑的地方(诚实边界)

- CUDA IMA 是**异步上报**的,栈里那句 "the stacktrace below might be incorrect" 是真的 ——
  触发 launch 的内核未必是真正越界的内核。两次都恰好落在**懒加载后 13–19 s** 与
  **prefill 负载**上,但这只是强相关,不是证明。
- 懒加载是**充分但非必要**:历史 `p2-t8p1` 88 次、`final150` 80 次告警都没崩;
  本会话 4 个实例里崩了 2 个(约 50%)。
- `git status` 干净、HEAD `d096e83b16`,**本轮未改 sglang 源码** → 这是**出厂配置的故障,不是我引入的**。

### 缓解方向(未验证)

1. **让启动期就覆盖 prefill 内核** —— 即开 prefill graph 或在 `mark_serving_started()`
   之前跑一次 prefill 前向。注意 `triton_load_watch.py:18` 明说 `--warmups` 与服务端 warmup
   都在 `mark_serving_started()` **之后**,不算数。
2. **`SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY=1`** 把每次晚期加载变成硬失败,
   一次性列全"预热没覆盖的特化"清单(代价:一旦不覆盖就起不来)。
3. 抬 PP3 余量(F3 层重排能把 PP3 的 11 层减到 10 层,**释放 1706 MiB/卡**)。

---

### 6.1 取证升级:阈值太严,日志只覆盖一个 stage

`triton_load_watch` 只在 `free_gb < SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB`
时才告警,默认 **1.0 GiB**(`environ.py:1256`)。本机只有 PP3(0.74 GB)低于阈值
→ **日志里只看得到 PP3 的特化上卡,其余 3 个 stage 在静默加载**。
把阈值设成 999 就是"每次上卡都打日志、每个 rank 都打",而钩子只在编译与首次加载时
触发(`triton_load_watch.py:16` 自述 steady-state cost is zero)→ **零成本、零行为改动的
取证升级,已从本轮起带在每个臂上**。

### 6.2 IMA 的取证结果:模式很硬,首因仍未定

**模式**:三次崩溃里 #1/#3(同 rank、同 `pp:device` PG)**IMA 前同一秒**都发生了
`_expand_prefill_causally_kernel` 的**第二次**特化上卡:

| | 该 kernel 首次上卡 | 第二次上卡 | IMA |
|---|---|---|---|
| #1 `crash_A_ima` | 00:30:23 | **00:30:42** | 00:30:42.543 |
| #3 `crash_M16_ima` | 01:21:19 | **01:21:32** | 01:21:32.000 |
| #2 `crash_A2_ima` | 1 次,**无第二次** | — | 前 **12 s 无任何上卡** |

该 kernel 的 `BS_P2 = next_pow2(bs)` 是 constexpr(`dsv4_attn_metadata_kernels.py:231`)
→ **换一次 bs 就编译并上卡一个新 cubin**。#3 里该 kernel 上卡 6 次 = 3 个事件 × TP2,
`_headshared_sparse_kernel` 三次崩溃都上卡 8 次(4 个事件)—— **特化数量本身就是变量**。

**两条被我自己证否的假设(记下来免得再走一遍)**:

1. **"prefill `else` 分支缺 `even_k` 守卫 → scale 越界读"**(`fp8_w8a16.py:605-619`)
   **不成立**。守卫确实只在 decode 分支有(`:504` 计算、`:597` 传入),prefill 分支确实没有
   —— 但本模型 `hidden_size=4096`、`moe_intermediate_size/tp=1024`、
   `weight_block_size=[128,128]`,**k 与 n 全是 128 的倍数**;逐维核过 `cdiv` 与
   `scale` 实际行/列数的关系(n 方向在最后一个 tile 也是安全的),**没有越界点**。
   → #2 的栈落在 `_dequant_block_fp8_kernel` 只是因为 CUDA 错误是**粘性**的,
   会在**下一个 launch** 处上报,那不是首因。
2. **"显存不够导致 `cuModuleLoadData` 失败"不是 IMA 的直接原因**:
   40 次上卡全部成功(0.79 → 0.61 GiB 只掉了 180 MB,远未耗尽),而且该模块讲的是
   **CUDA OOM**,现场是 **IMA**。→ **F3 放显存能消掉告警,但不构成 IMA 的修复**,
   这条要在 §9-B 里更正。

**仍然没有首因。** 要拿到它只能用 `CUDA_LAUNCH_BLOCKING=1`(让错误在真正出事的
launch 处上报,而不是粘到下一个),代价是全同步、吞吐掉一个量级 —— 但我们只需要
活到第一批请求即可。**这是下一个诊断臂。**

---

### 6.3 prefill CUDA graph 可行性探测:**显存不是障碍,是代码 bug(已修)**

**先纠正一个我自己的误判。** 我原以为 MLA 预留会挡死它 —— 实测**预留根本没生效**:
PG1 里 `available_bytes=0.31 GB`、`full_token=153600` 与基线**逐字相同**,PP3
`Memory pool end avail=0.92 GB`(还比基线 0.79 GB 多)。说明本模型走的是
`len(bs)*8` 那条分支(29 桶 = 232 MB),不是 `1.5*1024`。
**结论:`--cuda-graph-backend-prefill` 从来没有被显存挡过,是我没先测就下结论。**

**失败点**(归档 `res/crash_PG1_prefillgraph.log`):

```
Capture target prefill CUDA graph begin. backend=breakable, num_tokens=[4,...,512]
  0%| 0/30   ← 第一桶就是 512(_capture_one_stream 用 reversed())
deepseek_v4.py:748  assert output[:real_num_tokens].numel() == ret.numel()
AssertionError: Output tensor element mismatch: 16777216 != 8388608
```

**根因(TP 下 100% 必现,不是偶发)**,三段代码对上:

| 位置 | 事实 |
|---|---|
| `deepseek_v4.py:1010 _kernel_num_heads` | `attn_tp_size>1` 且 **非 sm120** → `return 64 if n_local_heads <= 64 else n_heads` → **本机恒 64** |
| `:826` | `n_local_heads = n_heads // attn_tp_size = 64/2 = **32**` |
| `:2276` | `kernel_num_heads != n_local_heads` → 建 `q_padded = [tokens, **64**, head_dim]`,`q_out = q_padded[:, :32, :]` |
| `:2391` | `attn_q = q_padded`(**padded 的 64**) |
| `:2393` 图分支 | `o = attn_q.new_empty((*attn_q.shape[:-1], v_head_dim))` → **64 头** |
| `:2373` unified 分支 | `q=q_out if q_out is not None else q` ← **用的是 local 的 32,是对的** |

于是 `output[:512] = 512×64×512 = 16777216`,backend 返回 `ret = 512×32×512 = 8388608`,
**比值恒 = 64/32 = 2 = tp_size,与桶无关**。捕获是倒序的(512 第一桶),所以
`0/30` 立刻崩 —— **不是"只有 512 失败",是每一桶都会失败**。

> **这不是显存/性能问题,是 breakable prefill graph 在 TP>1 下从未跑通过。**
> 也解释了为什么生产(`disabled`)永远碰不到它 —— 这条路径只在
> `is_in_breakable_cuda_graph()` 里才走。

**修复**(`deepseek_v4.py:2391` 图分支,1 处):输出缓冲的 head 轴改用
`self.n_local_heads`,`q_padded` 本身不动(它是给 kernel 的 padded 布局,不能改)。
比照 `unified` 分支的写法,语义就是"让图缓冲与 eager 路径 `attn_backend.forward`
的返回值同形" —— **这是该 op 自己 assert 声明的契约**。

⚠ 仓库因此**不再是干净的**(HEAD `d096e83b16` + 这一处 diff),要记在交付物里。

**PG2 结果:上一个断言过了,死在更深一层 —— 而且真首因被 PyTorch 的次生异常盖住了。**

```
# 链式异常的第一条(真首因):
breakable_cuda_graph.py:230  wrapper → _end_current_segment → graph.capture_end()
torch.AcceleratorError: CUDA error: out of memory            ← cudaErrorMemoryAllocation
# 第二条(__exit__ 对同一个、未被清掉的 graph 再 end 一次 → 计数器已 0):
RuntimeError: num_active_captures_ > 0 INTERNAL ASSERT ... markCaptureEnd called with no
captures in progress                                        ← 次生,纯误导
```

> **教训**:看到 PyTorch `INTERNAL ASSERT` 要往上找 **"During handling of the above
> exception"**。直接照它去查 capture 生命周期会白查一轮 —— 它只是
> `__exit__` 对已经失败的 `_current_graph` 重复 `capture_end`。

**各 stage 的捕获显存账**(归档 `res/crash_PG2_captureend.log`):

| stage | 池末 free | 捕获起点 | 512 桶跑完后 | 结果 |
|---|---|---|---|---|
| PP0 | 2.40 GB | 2.37 GB | 继续 | 过 |
| PP1 | 1.58 GB | 1.55 GB | **1.08 GB** → 进 480 | 过 |
| PP2 | 1.63 GB | 1.60 GB | 继续 | 过 |
| **PP3** | **0.93 GB** | 0.92 GB | **0.20 GB** → 进 480 即 **OOM** | **挂** |

- **单个 512 桶在 PP3 上耗 0.72 GB**(0.92 → 0.20),而且**它自己是成功的** —— 死在
  下一桶 480 的增量分配。
- 桶共享同一个 mempool,所以**只捕 512 一桶时 PP3 有 0.92 > 0.72 GB,理论能过**;
  而 replay 是**向上取桶**,512 正好覆盖 chunk=512 的主路径。
- 0.72 GB 对一个 512-token forward 来说大得离谱 → 怀疑是
  `--cuda-graph-prefill-max-context` 缺省时 **context-shaped attention metadata 与
  indexer logits 按模型最大上下文(262144)分配**(`fields/exec_.py` 的 help 原文)。

**PG3 配置**:`PREFILL_GRAPH_BS=512`(单桶)+ `PREFILL_GRAPH_MAX_CONTEXT=65536`。

> ⚠ 封顶 64k 意味着 **>64k 上下文会回落 eager** —— **128k/256k 长上下文臂不能带这两个变量跑**,
> 否则测出来的就不是 graph 了。这条要记进 §9。

**结论链(三层,每层都推翻了上一层的判断)**:

1. "prefill graph 被 MLA 1536 MB 预留挡死" → **错**,本模型走 `len(bs)*8` 分支,预留没生效。
2. "代码 bug,TP>1 下 `o` 的 head 轴错用 padded" → **真,已修**(`deepseek_v4.py` 图分支)。
3. "修完就能跑" → **还不够**,PP3 只有 0.92 GB,捕获本身 OOM。
   → 要么砍桶数/上下文(本轮 PG3),要么走 **F3 层重排给 PP3 放 1706 MiB**。
   **prefill CUDA graph 由此变成 F3 的下游依赖**,这是个新的量化理由(见 §9)。

**后续**:PG3 起来后跑 `measure.sh` 拿 prefill A/B + greedy parity(注意 graph 可能改变
padding 下的数值,parity 6/6 是硬闸)。

---

### 6.4 PG3:**捕获成功,但 prefill graph 消不掉 IMA —— 臂被熔断作废**

**PG3 配置**:`PREFILL_GRAPH=breakable` + `PREFILL_GRAPH_BS=512`(单桶)+ `PREFILL_GRAPH_MAX_BS=512`
+ `PREFILL_GRAPH_MAX_CONTEXT=65536`。

**捕获成功了**(`res/restart_PG3.log`,`8 begin / 8 end` = 4 stage × 2 TP):

| stage | 捕获起点 | 单桶 512 捕获用量 | 捕获后 avail |
|---|---|---|---|
| PP0 | 2.37 GB | 0.31 GB | 2.06 GB |
| PP1 | 1.57 GB | 0.35 GB | 1.25 GB |
| PP2 | 1.60 GB | 0.41 GB | 1.15 GB |
| **PP3** | 0.92 GB | **0.43 GB** | **0.49 GB** |

- **上下文封顶由效果反证生效**:同样是单个 512 桶,PG2 耗 **0.72 GB**(且 OOM)→ PG3 耗 **0.43 GB**,
  差 0.29 GB 只可能来自 `--cuda-graph-prefill-max-context 65536`。印证了"缺省按模型最大上下文
  262144 分配"那段 help 的判断。
- **KV 池完全没被挤占**:`full=153600/168448/209152` 与基线**逐字相同**。
- **但 measure 熔断**:`./measure.sh PG3` → `exit 3`,`[0..4] a=None / b=URLError:
  Connection refused`,`[5] IDENTICAL (0 chars)`(这是已知的"两侧都错 → 0 字符"误报)。
  **服务器在 greedy 阶段挂了 → prefill 吞吐根本没测到。** 日志已归档 `res/crash_PG3_replay.log`。

#### IMA 复现,而且这次带着全量取证

崩溃签名与 #1/#3 **完全同族**:`rank5 = PP2 TP1`、`PG ID 6 PG GUID 61(pp:device) Rank 2`、
11:44:12.587 `illegal memory access`。

阈值 999 下,每个 rank 每次上卡都记了日志 —— **四批 load 的对照表**:

| 时刻 | device-load 的 kernel | 是否 IMA |
|---|---|---|
| 11:41:56–57 | 14 种 × 8 rank(初始懒加载,24/24/24/4+8/6/6/26) | 否 |
| 11:43:23 | `_expand_prefill_causally_kernel` ×8(config 1) | 否 |
| 11:44:01 | `_headshared_sparse_kernel` ×16 | 否 |
| **11:44:12** | `_expand_prefill_causally_kernel` ×8(**config 2**) | **是** |

> **`_expand_prefill_causally_kernel` 被加载了三次,只有第三次崩。**
> → **Triton 上卡既不充分也非必要**,与 #2(前 12 s 无任何上卡照样崩)互相印证。
> 那条 `Pre-load it during engine init to avoid CUDA OOM` 的注释方向是错的:现场是 IMA 不是 OOM。

#### 否定 3:「prefill CUDA graph 一石二鸟」**被证否**

原本的期望是"启动期捕获 → 不再中途上卡 → IMA 消失"。实测 **prefill graph 开着,
serving 期上卡照旧**,而且不只 `_expand_prefill_causally_kernel`:

```
16 × alloc_decode_kernel          8 × alloc_extend_kernel        8 × compute_position_kernel
24 × _expand_prefill_causally_k.  32 × _headshared_sparse_kernel 8 × get_and_clear_swa_pages_kernel
 8 × _init_compressed_attn_md_k.  8 × _moe_align_small_numel_k.  8 × _mqa_paged_smallq_kernel
 8 × _page_table_positions_k.      2 × _small_copy_kernel         8 × _topk_transform_paged_k.
 8 × _wo_a_absorb_kernel           8 × write_req_to_token_pool_k.
```

原因很清楚:**breakable 的语义就是"断点处跳出 graph 跑 eager"**,注意力断点体
(`bcg_deepseek_v4_attention_with_output`)在**每次 replay 时都是 eager 调用**,于是
per-shape 的 Triton JIT 照样在服务期触发。**prefill graph 与"消灭服务期懒加载"解耦了。**

另注:`_headshared_sparse_kernel` 在服务期被加载 32 次 —— headshared 路径虽已列为
"勿再投入",但它是**活的**,只是被我们证否了收益,不是被关掉了。

**结论:prefill CUDA graph 现在只剩"prefill 提速"这一个预期收益(未测),
IMA 收益归零。** 取首因只能靠 `CUDA_LAUNCH_BLOCKING=1` → **LBO1 诊断臂**(见 §9)。

---

### 6.5 ~~IMA 首因:`_headshared_sparse_kernel` 越界~~ → **已被 §6.6 降级为"报告帧",非首因**

**方法**:`CUDA_LAUNCH_BLOCKING=1` 其余配置与 PG3 完全一致 → 重启 `healthy after 90s` →
greedy 前 5 个 prompt 全成功、**第 6 个 `RemoteDisconnected`** → 归档 `res/crash_LBO1.log`。

同步执行让错误在**真正出事的 launch 当场**报出,而不是粘到下一个 —— 于是拿到了三条
**Triton 自己抛的**栈(不再是 NCCL watchdog 的次生报错):

```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
  deepseek_v4.py:2419            o = attn_backend.forward(...)
  deepseek_v4_backend.py:3553    o = flash_mla_with_kvcache_sm120(...)
  flash_mla_sm120.py:470         flash_mla_sparse_decode_triton(...)
  flash_mla_sm120_triton.py:1540 return _run_headshared_sparse_decode(...)
  flash_mla_sm120_triton.py:1439 _headshared_sparse_kernel[grid](...)
```

- **两处独立上报**:`PP2 TP0` 与 `PP3 TP0` → 正好对上历史三次的
  `rank5 = PP2 TP1`、`PG 6 GUID 59/61 (pp:device) Rank 2`。**stage 一致,吻合。**
- 历史上"IMA 前同一秒有 Triton 上卡"的取证方向是**误导**:
  `_expand_prefill_causally_kernel` 只是恰好同秒被 JIT 编译;真凶是
  `_headshared_sparse_kernel`(PG3 里它在 11:44:01 ×16 上卡,11:44:12 崩)。
  **同秒相关 ≠ 因果,这是本轮最大的方法论教训。**
- 也解释了 §6.4 的"懒加载非充分条件"(#2 前 12 s 无上卡照样崩):崩的从来不是"加载时刻",
  而是**这个 kernel 被调用**时;加载与否只决定它是否恰好同时出现在日志里。

#### 开关与代价

`SGLANG_SM75_HEADSHARED_MIN_BATCH`(`environ.py:1163`,**默认 1**),注释:
`0 disables head-sharing entirely, which is the A/B switch`。

> ⚠ **不能长期关**:`environ.py:1148-1161` 记录 head-shared 现在是**赢家** ——
> 86K ctx / bs=1 / TP2 / PP4 / 150 W,**16 split 下 13.55 tok/s vs per-head 11.00 = +23%**,
> 且"the kernel now wins at every batch size"(阈值因此从 2 降到 1)。
> 关掉它 decode 会掉,所以它只是**诊断开关 + 临时解封**,真修法是找越界点。

**但它是 decode 专属**(`_run_headshared_sparse_decode`,栈里也只有 decode 路径),
**关掉不影响 prefill** —— 正好可以先把 prefill graph 的测量解封。

**PG4 臂(已启动)**:`HEADSHARED_MIN_BATCH=0` + `PREFILL_GRAPH=breakable`(其余同 PG3),
一石二鸟:① 若整轮 measure 无 IMA → **首因坐实**;② 拿到一直没测成的 prefill A/B。

> ⚠ **PG4 的 decode 数与 A / M16 不可比**(headshared 状态不同);
> prefill 数**可比**(headshared 不参与 prefill)。跨臂比对时必须注明。

#### 下一步:真修法

1. **确认**(PG4):关掉后 IMA 消失 → 首因坐实。
2. **定位越界点**:`_headshared_sparse_kernel`(`flash_mla_sm120_triton.py:667`)里嫌疑有三
   —— ① `indices` 容量填充区里的垃圾页号 ≥ `num_pages`;② `topk_len` 掩码漏配;
   ③ `cache_u8_ptr`/`page_bytes` 的字节偏移(`as_strided + view(torch.bfloat16)` 重解释)。
   首选做法是**把 `_run_headshared_sparse_decode` 抽成独立 repro,用
   `compute-sanitizer --tool memcheck` 打到具体指令**,而不是继续在日志里找相关性。
3. 修好后把 `MIN_BATCH` 恢复 1,拿回那 +23%。

---

### 6.6 **headshared 被证否;重新定性:真实错误在 CUDA graph replay 里**

**PG4 臂** = `HEADSHARED_MIN_BATCH=0`(headshared 关)+ `PREFILL_GRAPH=breakable`,其余同 PG3。

| 检查 | 结果 |
|---|---|
| headshared 是否真关掉 | ✅ `res/crash_PG4.log` 里 `_headshared_sparse_kernel` **上卡 0 次**(LBO1/PG3 是 32/16 次) |
| 还崩吗 | ✅ **照样崩**,greedy 第 1~2 个 prompt 就 `RemoteDisconnected` |
| 栈变成什么 | `fp8_w8a16.py:617 _w8a16_linear_impl` → `:424 dequant_block_fp8_slice` → `_dequant_block_fp8_kernel[grid]` |
| 上报 rank / PG | `rank4`+`rank5`(PP2),`PG ID 2 PG GUID 7(**tp**:device) Rank 0/1` |

> **`SGLANG_SM75_HEADSHARED_MIN_BATCH=0` 不消除 IMA → §6.5 的"首因"结论不成立。**
> 它只是这一轮**恰好被报出来的那个 eager launch**。

#### 三处证据一起把结论改写

1. **栈帧随配置漂移**:headshared 开 → headshared;headshared 关 → fp8 dequant。
   真正的首因不会因为关掉某个旁观者就换一个。
2. **但各自内部两 rank 完全一致**(LBO1:PP2 TP0 + PP3 TP0 同帧;PG4:PP2 两 rank 同帧)
   → 不是随机噪声,是"**该配置下 replay 之后的第一个 eager launch**"稳定地背锅。
3. **上报的 PG 类型都不同**:LBO1 是 `(pp:device) GUID 61`,PG4 是 `(tp:device) GUID 7`。
   → 哪个 watchdog 先说话取决于各 rank 下一次 collective,**这正是异步/无帧错误的特征**。

#### 假设:错误发生在 **decode CUDA graph replay** 内部

replay 是**一次 `cudaGraphLaunch`,没有 Python 帧**,所以 CUDA 的异步 IMA 必然被
**之后第一个真正进 Python 的 Triton launch** 记下 —— 这解释了全部三个现象
(换配置就换栈、同一配置内两 rank 同帧、PG 类型随机)。

**恒定的共同点**(至今所有崩溃无一例外):

| 臂 | prefill graph | headshared | **decode graph** | 结果 |
|---|---|---|---|---|
| A / A2 / M16(生产) | off | on | **full** | 崩 3/5 |
| PG3 | breakable | on | **full** | 崩 |
| LBO1 | breakable | on | **full** | 崩 |
| PG4 | breakable | **off** | **full** | 崩 |
| **LBO2** | off(生产) | on | **disabled** | **← 正在跑** |

**LBO2 是单变量实验**:生产配置 + 只关 `--cuda-graph-backend-decode`。
- 若**整轮 measure 不崩** → 坐实"错误在 decode graph 内",下一步在 graph 里二分;
- 若**照样崩** → graph 假设也倒,回退到"eager 路径多处真越界",那时
  **`CUDA_LAUNCH_BLOCKING` 对 Triton(driver API `cuLaunchKernel`)是否真生效**必须先验证。

> ⚠ 方法论:`CUDA_LAUNCH_BLOCKING` 官方只讲 **CUDA Runtime**;Triton 走 **driver API**。
> LBO1 里错误确实是从 `driver.py:328 self.launch()` 自己抛出来的(说明同步生效了),
> 但 PG4 在**没开** blocking 的情况下也从同一个 `self.launch()` 抛出 → **单凭"从 launch 抛出"
> 无法区分同步报错与异步残留**。这条必须在结论里保留不确定性。

---

### 6.7 **三连证否:Python 栈不可信 → 换 driver 级 CUDA coredump(CD1 臂)**

#### 六份 IMA 档案的横切面

用 `res/crash_*.log` 统一提取"`Triton Error [CUDA]` 往上最近的 sglang 帧":

| 档案 | 配置差异 | 有 Python 栈? | 报告帧 | 上报 rank / PG |
|---|---|---|---|---|
| `A2` | **纯生产** | ✅ | `fp8_w8a16.py:424 dequant_block_fp8_slice` | — |
| `PG4` | headshared **关** | ✅ | `fp8_w8a16.py:424 dequant_block_fp8_slice` | rank4/5,`PG 2 GUID 7(tp:device)` |
| `LBO1` | **`CUDA_LAUNCH_BLOCKING=1`** | ✅(×2) | `flash_mla_sm120_triton.py:1439 _run_headshared_sparse_decode` | PP2 TP0 + PP3 TP0,`PG 6 GUID 59/61(pp:device)` |
| `LBO2` | decode graph **关**(0 次捕获已确认) | ✅ | `elementwise.py:649 moe_swiglu_clamp` | rank6/7,**PP3** |
| `A` / `M16` | 纯生产 | ❌ 只有 NCCL watchdog | — | — |
| **`PG3`** | **= LBO1 配置,但 blocking 关** | **❌ 只有 NCCL watchdog** | — | — |

**三处硬结论:**

1. **四个互不相干的报告帧**(`dequant` ×2 / `headshared` / `swiglu`),四个里三个是
   attention / 量化 / 逐元素,**没有任何两个属于同一个子系统** → 它们不是"各自越界",
   是"**同一个粘性错误,落到当时第一个 eager Triton launch 上**"。
   (A2 与 PG4 恰好同帧,只是因为两者生产路径的 launch 顺序相近。)
2. **`PG3` vs `LBO1` 是天然受控对照**:同配置、唯一差别是 blocking。
   PG3 **没有** Python 栈,LBO1 **有** → **blocking 确实改变了错误上报时机**。
   但它给的仍是"下一个 launch"(**LBO1 的 headshared 已被 PG4 关掉后照崩证否**),
   **只改时机、不改可信度**。
3. **LBO2 证否 decode graph**:decode graph 关掉(日志 `Capture target decode CUDA graph`
   计数 = 0)**照样崩**,且帧换到 `moe_swiglu_clamp`、rank 从 PP2 漂到 PP3。
   → §6.6 的"错误在 replay 内部"假设**也倒了**。

#### 至此已证否的 IMA 假设清单

| # | 假设 | 证否方式 |
|---|---|---|
| 1 | 服务期 Triton **懒加载**导致 IMA | 同秒相关非因果;#2 案例前 12 s 无上卡照样崩;同一 kernel 加载 3 次只第 3 次崩 |
| 2 | prefill CUDA graph 能"顺带消 IMA" | §6.4:breakable 断点体每次 replay 都跑 eager,服务期上卡照旧,PG3/PG4 照崩 |
| 3 | **`_headshared_sparse_kernel` 越界是首因** | PG4:0 次 headshared 上卡(开关确实生效)→ 照崩,帧换成 `dequant` |
| 4 | **错误发生在 decode CUDA graph replay 内** | LBO2:decode graph 关掉(0 次捕获)→ 照崩,帧换成 `swiglu` |
| 5 | `CUDA_LAUNCH_BLOCKING` 能给出可信首因 | 只改时机不改可信度;且它只管 Runtime API,Triton 走 `cuLaunchKernel` |

#### CD1 臂:driver 级 coredump(唯一还没试过的可信信源)

sglang 内置(`python/sglang/srt/debug_utils/cuda_coredump.py`,`environ.py:395`):

```
SGLANG_CUDA_COREDUMP=1                 # 注入 CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
SGLANG_CUDA_COREDUMP_DIR=<workspace>/res/coredumps
SGLANG_CUDA_COREDUMP_BEFORE_CRASH=1    # 默认开,tokenizer_manager 收到异常时再让各 rank 落盘
```

- dump 由 **CUDA driver 在设备异常现场生成**,与 Python 栈 / NCCL watchdog / 线程竞速**无关**
  → 是这条链上唯一不受"粘性错误"污染的信源。
- `_CUDA_COREDUMP_FLAGS` 默认 `skip_global_memory,skip_shared_memory,...` →
  **轻量、不含内存内容**,但**保留 kernel 名 + PC + 寄存器**,足够定位"哪个 kernel、哪条指令"。
- 分析:`/usr/bin/cuda-gdb -c <dump>` → `info cuda kernels` / `bt` / `disas $pc`。
- 环境变量经 `os.environ` 注入,子进程自动继承 → 8 个 rank 各落各的文件
  (`%h.%p.%t` = host/pid/时间,不冲突)。

**已知残余风险**:若首因是 *host 侧*(如野指针写坏 CUDA context),coredump 不会生成;
那时它会以 `CUDA_COREDUMP_SHOW_PROGRESS` 的失败信息暴露出来,也算信息。

**CD1 臂 = 纯生产配置 + 只开 coredump**,跑 `restart + greedy`(崩溃固定发生在
greedy 前 1~6 个 prompt,不需要跑完整 measure)。

#### CD1 结果:dump 拿到了且完整,但 **gdb 全系读不了**(工具链代差)

- **dump 成功且写完整**:2 份各 **199,755,747 B**(PID 2498266/2498267),
  日志 `coredump: All done (took 01s)`,且文件大小与 ELF 最大 section 末端**精确吻合**
  → 不是截断。进度里能看到 `SMs 0..67 are not used by any context`(仅本 rank 的
  device 0 有 context)。
- **gdb 全部拒绝**:`"<dump>" is not a core dump: file format not recognized`。
  ELF 头是 `Machine: NVIDIA CUDA architecture (0xbe)`、**`e_phnum = 0`(零个 program header)**。
  依次试过 **4 个**:system `cuda-gdb 12.0`、`/usr/local/cuda-12.9/bin/cuda-gdb 12.9`、
  **新装 `cuda-gdb-13-2` 13.2.86**(`apt-get download` + `dpkg -x` 解到 `/tmp/opencode/cuda13`,
  版本对上 driver 的 CUDA 13.2)、系统 `gdb` → **全拒**。
- **不是 CD1 特有 → 先前的假设证否**:写了个 20 行的最小复现(裸 Triton 越界
  `tl.store`,确认真的 IMA),生成的 dump **同样 `phnum=0`**;而且
  **`noskip`(完全不设 `CUDA_COREDUMP_GENERATION_FLAGS`)也一样** →
  ~~"是 sglang CI 的 `skip_*` flags 把 PT_LOAD 砍光了"~~ **证否**。
- **`log_only` 环境变量没生效**:cuda-gdb 文档明写
  `log_only` = *"Do not generate a corefile, only log the exception information
  (**PC, stack trace, kind, etc.**). Use `CUDA_LOG_FILE`"*,但
  `CUDA_COREDUMP_GENERATION_FLAGS=log_only` **照样写了 13.6 MB corefile**,
  `log.txt` 里 **0 条**异常信息(只有 `cuCtxGetDevice` 的 `INVALID_CONTEXT` 噪声)。
- **好消息:dump 里符号是全的**。`strings` 能取到 `.text.oob_store` / `.nv.info.oob_store`
  / `.nv.shared.oob_store`,并且还有 `.cudbg.lntbl`(逐 warp 行表 1792 B)、
  `.cudbg.regs` / `.cudbg.pred`(逐 warp 逐 lane)、`.cudbg.relfi`(relocated ELF,生产 dump 127 份)。
  → **可解析,但** `gridtbl`/`ctxtbl`/`ctatbl` 的 size 为 0,要反推格式;
  投入产出比不如换信源。
- **两个格式坑**:`e_shnum = 0` 是 **SHN_XINDEX**(真值在 section #0 的 `sh_size`,
  必须特判,否则 `struct.unpack` 直接崩);`e_machine` 非 x86-64,system gdb 也会拒。

#### SY1 臂:**launch 后立即同步**,把粘性错误钉回真凶(不依赖 gdb)

- **机制根因**:`CUDA_LAUNCH_BLOCKING` 只管 **CUDA Runtime API**,而 Triton 走
  **driver API `cuLaunchKernel`**(`driver.py:328`)→ 异步 IMA 总是报给
  "**下一个** eager launch",这正是四份档案四个互不相干栈的成因(§6.7 横切面)。
- **做法**:在 `CudaLauncher.__call__` 返回后**立即 `torch.cuda.synchronize()`** →
  错误**最多晚一个 launch** 就会抛出,而不是晚几千个。
- **改动(3 处,全部 env 门控、默认关,不影响生产)**:
  1. `environ.py`:新增 `SGLANG_TRITON_SYNC_EVERY_LAUNCH = EnvBool(False)` + 注释;
  2. `triton_load_watch.py`:新增 `install_launch_sync()` / `_sync_after_launch()` —
     只在 **stream capture 期间跳过**(capture 里 `cudaDeviceSynchronize` 非法),
     其余异常吞掉但**放行** `illegal memory access` / `unspecified launch failure` /
     `invalid device function` / `misaligned address` / `assertion`;
  3. `scheduler.py` `run_event_loop()`:`install()` 旁边调 `install_launch_sync()` —
     **在 engine init(graph capture + warmup)完成之后**,天然避开 capture。
- **为什么比"只同步 Triton launch"更强**:它是 **device-wide** sync,所以任何先前的
  torch kernel 出错也会在**紧跟的那次 sync** 报出 → 上界是"晚一个 launch"。
- **代价**:设备完全串行。启动/健康检查不受影响(init 在装钩之前),
  但 **greedy 会慢很多**,该臂的**吞吐数据一律作废**。
- **本臂配置**:纯生产 + `SGLANG_TRITON_SYNC_EVERY_LAUNCH=1`,coredump **关掉** ——
  coredump 的默认行为(`no_errbar_at_exit` 反向)会给 kernel 加 error barrier
  **改变执行行为**,诊断臂不该带。
- 产物:`res/restart_SY1.log`、`res/greedy_SY1.log`、`res/serve_SY1.log`。

#### SY1 结果:**可信栈到手** → `_headshared_sparse_kernel`

钩子确实生效(每个 rank 都打了同步告警),greedy 第 5 个 prompt 崩,栈:

```
flash_mla_sm120_triton.py:1439  _headshared_sparse_kernel[grid](
  :1540  _run_headshared_sparse_decode   ←(经 _run_sparse_attention 分派)
  :474   flash_mla_sparse_decode_triton
flash_mla_sm120.py:470  flash_mla_with_kvcache_sm120
deepseek_v4_backend.py:3553  flash_mla_with_kvcache_sm120
deepseek_v4.py:2419  attn_backend.forward
eager_runner.py:371  _execute_extend        ← 发生在 extend(prefill)前向里
torch.cuda.synchronize() → torch.AcceleratorError: illegal memory access
```

受害者 `[rank5]` = **PP2 TP1**。因为是 launch 后**立刻 device-wide sync**,
栈的归因上界是"晚一个 launch",而 launch 就在 sync 正上方 → **可信**。

#### PG5:**所谓"第二个坑"其实是同一个坑**

PG4 那次"关掉 headshared 照样崩"曾指向第二个 bug,先复核了它的可信度:

- `restart_*.log` 的 flags 回显是**从 `/proc/<pid>/environ` 直读**的,但过滤模式
  `^SGLANG_SM75_` 是 **12:17(CD1 那轮)才补进去的**,PG4 跑在 11:58 → **不能用回显判断**;
- 改用**不依赖日志格式**的判据:PG4 开服后 1.5 分钟一共懒加载 **114 个** Triton kernel,
  `_headshared_sparse_kernel` **0 个**;而 headshared 开着的臂(LBO1/CD1/SY1)全是 **33–34 个**,
  且这些上卡告警都发生在**开服后 2 秒**(不是崩溃前凑出来的)→ **开关确实生效**。
- PG4 错误类型同样是 `illegal memory access`,受害者 `rank4/rank5` = PP2 → 确实是同一个故障家族。

于是跑了 **PG5 = 关 headshared + 同步钩子**,拿到**第二份可信栈**:

```
flash_mla_sm120_triton.py:271   _tiled_sparse_decode_kernel[grid]     ← 换成它了
  :1543  _run_sparse_attention → _run_triton_sparse_decode
  :474   flash_mla_sparse_decode_triton
flash_mla_sm120.py:470  flash_mla_with_kvcache_sm120
eager_runner.py:371  _execute_extend
```

flags 回显**这次有** `SGLANG_SM75_HEADSHARED_MIN_BATCH=0`,服务期 headshared 上卡 **0 次**。

> **⇒ 不是两个 bug,是一个。** SY1 和 PG5 都在 `flash_mla_sparse_decode_triton` 里,
> 只是分派到了不同 kernel。**"栈随配置漂移"的全部现象 = 选中哪个 kernel 就崩在哪个。**

#### 根因(源码审读)

两个 kernel 的寻址方式完全一样,且**都缺上界**:

```python
safe = tl.where(idx_valid, raw, 0).to(tl.int64)   # idx_valid = (t_idx < valid_len) & (raw >= 0)
page_ids = safe // page_size
tok_base  = page_ids * page_bytes + page_offs * _HS_TOKEN_BYTES   # 直接当地址
```

- `_tiled_sparse_decode_kernel`:`flash_mla_sm120_triton.py:146-148`(同款 `raw >= 0` only)
- `_headshared_sparse_kernel`:`flash_mla_sm120_triton.py:800-802` + `_hs_load_nope:618`
- host 侧两处都算出了 `num_pages = k_cache.shape[0]`
  (`_run_triton_sparse_decode:238` / `_run_headshared_sparse_decode:1397`),
  **却从未传进 kernel** → kernel 物理上无法校验 `raw < num_pages * page_size`
- `indices` 是 int32,`page_ids` 转 int64 → 无溢出,但一个大值就把 `tok_base`
  直接推出缓存末尾 → IMA

**两条路(`_headshared_*` 与 `_tiled_*`)、两处缺上界 → 一个根因。**

#### SV1:校验器(正在跑)

`SGLANG_VALIDATE_SPARSE_INDICES=1` → 在唯一入口 `flash_mla_with_kvcache_sm120`
做主机侧上界检查:`indices >= num_pages*page_size` 就 **打 `SPARSE_INDICES_OOB`
日志 + 夹紧**(返回副本,不改调用方的 page-table 张量)。同步钩子同时开着。

- 若 **`SPARSE_INDICES_OOB` 命中且不再 IMA** → 根因坐实,可据此定真修
  (kernel 内加 `num_pages` 上界 mask,或 host 侧夹紧);
- 若 **不命中却仍 IMA** → `indices` 干净,问题在别处(那就只剩 `extra_k_cache` /
  别的地址源),同步钩子继续给栈。

#### SV1 → SV2:**校验器第一次启动失败**(可复用的坑)

`SGLANG_VALIDATE_SPARSE_INDICES=1` 用 `bool((indices >= ntok).any())` 做检查,
这会**同步设备**;而它跑在 **decode CUDA graph 捕获期间** →
`cudaErrorStreamCaptureUnsupported: operation not permitted when stream is capturing`
→ **graph 捕获被作废 → 服务起不来**(`restart exit=1`,端口 Connection refused)。

⚠ 教训:**任何诊断代码里带 `.item()` / `bool()` / `.any()` 的,都必须先判
`torch.cuda.is_current_stream_capturing()` 再跳过。** 同步钩子当时加了这个保护,
校验器漏了 —— 一并补上后重跑为 **SV2**。

> 顺带:`match_num_queries` 对 `swa_page_indices`/`swa_topk_lengths`/`extra_indices`
> 的补行值是 `0`/`1`/`-1`,都是**合法或被掩掉**的值 → 补位本身不会造出越界,
> 所以越界若存在,只可能来自上游 page-table 本身。

产物:`res/restart_SV1.log`(启动失败)、`res/restart_SV2.log`、`res/greedy_SV2.log`、`res/serve_SV2.log`。

#### SV2 结果:**根因坐实** —— IMA 消失,greedy 6/6 全过

```
[PP2 TP0] SPARSE_INDICES_OOB [extra]: 1 of 7168 entries outside [0, 38656), max=811112383
[PP2 TP1] SPARSE_INDICES_OOB [extra]: 1 of 7168 entries outside [0, 38656), max=811112383
[PP2 TP0] SPARSE_INDICES_OOB [extra]: 1 of 8192 entries outside [0, 38656), max=678864575
[PP2 TP1] SPARSE_INDICES_OOB [extra]: 1 of 8192 entries outside [0, 38656), max=678864575
```

四件事同时对上:① 受害者 **PP2 TP0/TP1** = 历史 `rank4/rank5`;② **7168 条里只有 1 条**脏;
③ `max` 远超池子(38656);④ **夹紧后无 IMA、greedy 6/6 跑完** → 因果闭合。
(⚠ flags 回显里看不到 `SGLANG_VALIDATE_SPARSE_INDICES` —— 过滤模式没加它 —— 但日志已证明生效。)

#### 元凶:`_topk_transform_paged_triton_kernel` 缺 `page_id` 上界

`extra_indices` = `c4_sparse_page_indices`(压缩 KV 的 page index),由
`kernels/ops/attention/dsv4/topk.py` 的 `_topk_transform_paged_triton_kernel` 写入:

```python
page_ids = raw // PAGE_SIZE
page_ids = tl.where(valid, page_ids, 0)
pages    = tl.load(page_tables_ptr + row * stride_page_tables + page_ids, mask=valid, other=0)
page_indices = pages * PAGE_SIZE + raw % PAGE_SIZE
```

- **`page_table_width` 是入参,函数体一次都没用**(全 kernel 只出现在参数表里);
- `raw` 只保证 `< seq_len`,**不保证 `raw // PAGE_SIZE < page_table.shape[1]`**;
- **脏值反推**(PAGE_SIZE=64):`811112383 = 12673630*64 + 63`、
  `678864575 = 10607258*64 + 63` → `rem` 都是 63,而 **`pages` = 千万级 page id,
  是不可能的值** → **读到了 `page_table` 之外**;
- `dsv4/metadata.py` 注释写明预期语义
  `page_indices[i, j] = page_table[i, j // page_size] * page_size + (j % page_size)`
  → `j // page_size` **本就必须**落在 table 宽度内。

#### 修复(SV3 验证中)

`topk.py` 把界补上:

```python
in_table = valid & (page_ids >= 0) & (page_ids < page_table_width)
pages    = tl.load(..., mask=in_table, other=0)
page_indices = tl.where(in_table, page_indices, -1)
```

越界 → `-1` → 被 attention kernel 既有的 `raw >= 0` 掩掉(语义:该槽位无效),
**不再产出脏地址**;开销只有一次比较。

**验证判据**:SV3 跑 `SPARSE_INDICES_OOB` 应从 **4 → 0**。
若仍 >0,说明 `page_table` 本身带**界内**脏值(机制 B),得改查 `page_table` 来源。

产物:`res/restart_SV3.log`、`res/greedy_SV3.log`、`res/serve_SV3.log`。

#### SV3:**修复生效** —— OOB 4 → 0,greedy 6/6,无 IMA

- flags 回显模式当时**还没**补 `SGLANG_VALIDATE_SPARSE_INDICES`(启动后才改的),
  所以先**直接读 `/proc/<pid>/environ` 确认 `SGLANG_VALIDATE_SPARSE_INDICES=1` 在**
  → 校验器**每一步都在查**,0 次不是"没开所以没报";
- `SPARSE_INDICES_OOB` **4 → 0**,全程无 IMA,greedy **6/6 跑完**。

⇒ **根因链闭合**:`page_id` 缺上界 = `extra_indices` 的唯一污染源。

#### 本轮仓库改动清单(HEAD `d096e83b16`,6 文件)

| 文件 | 性质 | 内容 |
|---|---|---|
| `kernels/ops/attention/dsv4/topk.py` | **★ 真修(无条件生效)** | `_topk_transform_paged_triton_kernel` 补 `page_id < page_table_width` 上界(越界写 `-1`) |
| `kernels/ops/attention/flash_mla_sm120.py` | 诊断 | `_validate_sparse_indices`:host 侧上界检查 + 夹紧,带 capture 保护 |
| `srt/environ.py` | 诊断开关 | `SGLANG_TRITON_SYNC_EVERY_LAUNCH`、`SGLANG_VALIDATE_SPARSE_INDICES`(均默认 `False`) |
| `srt/utils/triton_load_watch.py` + `srt/managers/scheduler.py` | 诊断开关 | launch 后 device-wide sync,带 capture 保护,`run_event_loop` 里装(在 engine init 之后) |
| `srt/models/deepseek_v4.py` | 既有脏文件 | 图分支 `o_shape` 修复(PG3 捕获用) |

⇒ **诊断开关默认全关,生产行为只多 `topk.py` 里那一次比较。**

#### 关掉诊断、恢复测量

`measure.sh <TAG>` 是对**当前活着的服务**跑全套(greedy×2 + decode + prefill),
所以流程 = `ab_restart.sh`(换 flag)→ `measure.sh <TAG>` → `ab_compare.py <A> <B>`。

⚠ **老 A 基线(10-01 00:43)与 PG3(11:45)是不同会话、不同构建**,按交错两臂规矩
必须**同构建重测**;旧产物已归档到 `res/archive_pre_fix/`。

PG3 臂 flag(记自 §6.4 与 `restart_PG3.log` 第 5 行):

```
PREFILL_GRAPH=breakable PREFILL_GRAPH_BS=512 PREFILL_GRAPH_MAX_BS=512 PREFILL_GRAPH_MAX_CONTEXT=65536
```

⚠ **`restart_*.log` 的 flags 回显不含 `PREFILL_GRAPH*`**(过滤模式没加)——
和 `SGLANG_SM75_*` / `SGLANG_VALIDATE_SPARSE_INDICES` 是同一类坑:
**别再靠回显判断没进模式的变量,要判就去读 `/proc/<pid>/environ`。**

#### A 臂(修复后纯生产)✅ 全绿 —— IMA 归零,parity 通过,无性能回退

`res/restart_Afix.log` / `res/measure_A.log` / `res/serve_Afix.log`(诊断开关全关)

- **IMA 0 次** —— 同配置在修复前近乎必崩(4/4 臂连崩)
- **greedy parity:6/6 byte-identical**,within-instance determinism **OK**
- **decode**(decode-graph 覆盖 **100.0%**,711/711):

  | bs | 1 | 2 | 4 | 8 |
  |---|---|---|---|---|
  | 修复后 | 26.7 | 41.6 | 70.4 | **107.3** |
  | 旧基线 A | 26.5 | 41.3 | 67.6 | 105.6 |

  → **无回退**(spread 分别 2.0% / 19.4% / 4.6% / 10.3%)

- **prefill**(aggregate tok/s,中位;`res/pf_A.json`):

  | lens | K=1 | K=2 | K=4 | K=8 | K8/K1 |
  |---|---|---|---|---|---|
  | 1000 | 795.5 | 1035.3 | 1209.8 | 1326.9 | **1.67×** |
  | 4000 | 1207.4 | 1327.3 | 1398.1 | 1435.5 | 1.19× |

- 旧 A / 旧 PG3 产物归档在 `res/archive_pre_fix/`(不同会话、不同构建,不可直接比)

**待办**:PG3 臂(同构建重测)→ `ab_compare.py A PG3` → F3/EAGLE 显存链。

---

### 6.8 prefill graph 吞吐 A/B:**`breakable` 慢 2.9×、`full` 未实现 → 整类关闭**

IMA 修复(§6.7)之后立刻补上这笔欠账。**两臂同构建、同脚本、同载荷历史**(老 A / 老 PG3
是不同会话不同构建,已归档到 `res/archive_pre_fix/`,不再引用)。

| 臂 | env | 状态 |
|---|---|---|
| **A** | `PREFILL_GRAPH=disabled`(= 生产),诊断开关全关 | ✅ IMA 0、parity 6/6 |
| **PG3** | `PREFILL_GRAPH=breakable` + `BS=512` + `MAX_BS=512` + `MAX_CONTEXT=65536` | ✅ 捕获成功,4756 个 prefill batch 走 graph |
| **PGF** | 同上但 `PREFILL_GRAPH=full` | ❌ **启动即崩**,见下 |

#### PG3 vs A(`ab_compare.py A PG3`)

| | A | PG3 | ratio | spread(A / PG3) |
|---|---|---|---|---|
| **prefill** lens1000 K=1/2/4/8 | 795.5 / 1035.3 / 1209.8 / 1326.9 | 273.4 / 353.7 / 416.4 / 453.7 | **0.344 / 0.342 / 0.344 / 0.342** | 1.6–3.7% / 4.7–6.7% |
| **prefill** lens4000 K=1/2/4/8 | 1207.4 / 1327.3 / 1398.1 / 1435.5 | 438.5 / 471.2 / 496.2 / 505.4 | **0.363 / 0.355 / 0.355 / 0.352** | 1.6–2.9% / 2.8–3.5% |
| **prefill 合计** | 1267.3 | 466.6 | **0.368** | — |
| decode bs1/2/4/8 | 26.7 / 41.6 / 70.4 / 107.3 | 26.5 / 42.1 / 70.4 / 108.7 | 0.993 / 1.012 / 1.000 / 1.013 | — |

- **prefill 慢 2.9 倍,8/8 格子全部 0.34–0.36**,而 noise 只有 2.8–6.7% → **效应是噪声的 10 倍**;
- **decode 完全不动**,**bs=1 控制项 0.993** → 两臂都健康,不是热漂移;
- **K 的 scaling 一模一样**(1.67× vs 1.66×)→ **不是调度问题,是每个 chunk 的恒定开销**;
- 已排除"chunk 被切小":两臂 `chunked_prefill_size` 都是 **512**(server_args 回显核对)。

**机制与源码里的两条既有告警吻合:**

1. `arg_groups/cuda_graph_hook.py:401`(DeepSeek-V3 + trtllm_mla 的先例):
   *"**a captured prefill graph forces a FlashAttention fallback that regresses prefill**"*
   → 现象类别一致:graph 确实在用(4756 次),却被逼到慢路径上;
2. `cuda_graph_hook.py:129`:**PP>1 时 `breakable` 默认就是关的**
   (*"enabling it implicitly would also capture large buckets that can be slower than eager"*)
   → 我们 pp=4,是**显式 opt-in** 才开的。

#### PGF:`full` 后端没为 dsv4 实现

8 个 rank 全部死在 capture:

```
prefill_cuda_graph_runner.py:1630  attn_backend.init_forward_metadata_out_graph(in_capture=True)
deepseek_v4_backend.py:2194        bucket = _GraphBucket.of(forward_batch.forward_mode)
deepseek_v4_backend.py:1107        raise NotImplementedError(...)
NotImplementedError: unsupported forward_mode=<ForwardMode.EXTEND: 1>
```

- `_GraphBucket` 只有 `DECODE_OR_IDLE / TARGET_VERIFY / DRAFT_EXTEND`,**没有 EXTEND 桶**;
- `prefill_cuda_graph_runner.py:1629` 明确分叉:`_is_full_backend` 走
  `init_forward_metadata_out_graph`,否则走 `_init_forward_metadata_for_capture`
  —— **dsv4 只实现了后者**;
- dsv4 的钩子命名本身就说明了:`init_forward_metadata_for_breakable_cuda_graph_capture` /
  `prepare_forward_metadata_for_breakable_cuda_graph_replay`;
- **不是参数问题,没有配置能绕过。**

#### greedy parity 附带发现

PG3 臂 `measure.sh` 报 `PARITY FAIL 1/6`(臂内两次在 prompt 0 不一致)。把 prompt 0 放进
**全部 27 份历史 greedy 跑**里分组:

| 变体 | 次数 | prefill graph ON | eager |
|---|---|---|---|
| v1 `391. 391 * 29 = 11339...` | 20 | 2(OLD:PG31、新 PG32) | **18** |
| v2 `391. 391 * 4 = 1564...` | 3(**新 PG31、PG41、PG42**) | **3** | **0** |

- **eager 臂 18/18 全是 v1;3 次 v2 全部来自 prefill-graph 臂**;
- 但同为 prefill-graph 的 OLD:PG31 与新 PG32 落在 v1 → **臂内也不自洽**;
- prompt 0 是近平局 prompt(模型自己挑乘数),对数值微差极敏感;
- §7 已记录"负载历史会改变贪心输出"(A0 是已知异类),prompt 2/5 的历史分叉确实由 A0 贡献,
  **与本项无关**。

**定性**:prefill graph 确实改变了 logits 数值,足以翻转近平局 token;3/5 vs 0/18 **方向一致
但样本小**,不单独作为定罪依据 —— **性能 −65% 才是主要否决理由。**

#### 结论

| backend | 状态 |
|---|---|
| `breakable` | 捕获成功,但 **prefill 0.368×**,否决 |
| `full` | **dsv4 未实现**,启动即崩 |
| `tc_piecewise` | **未测**;它不走 `init_forward_metadata_out_graph`(走 `get_tc_piecewise_forward_context`,dsv4 有钩子)→ 技术上可能可用,但要 torch.compile 捕获,成本高 |
| `disabled` | **保持生产值,生产无需改动** |

---

### 6.9 F3 层重排 `11,11,11,10`:**prefill +7.6%,建议上线**

**先说为什么重开一个"已关闭"的实验。** `PREFILL_PROFILE.md:665` 有一节
《11,11,11,10 为什么不能上线:PP0 的显存是硬约束》,结论是**不可用**。差点直接引用它 ——
核数字时发现**今天的余量已经不是当年那个数**:

| PP0 decode graph 捕获后余量 | 9/29 记录 | 本轮实测 |
|---|---|---|
| 现切分 `10,11,11,11`(10 层) | 0.74 GB | **0.93 GB** |
| F3 `11,11,11,10`(11 层) | **0.42 GB → GSM8K 崩** | **0.64 GB** |

崩溃阈值只知道落在 **0.42(崩)~ 0.74(安全)** 之间,**0.64 正好落在未知区间** →
不能引用旧结论,必须实测。

#### 机制(`PREFILL_PROFILE.md:608`)

43 层分给 4 个 stage,余数给靠后的 stage,所以现切分是 `10,11,11,11`。实测每 stage 每 chunk:
PP0 215.2 ms / PP1 254.8 / PP2 247.4 / **PP3 274.6** —— **PP3 是瓶颈**(它每层比 PP0 贵 16%,
还要背 lm_head + 采样)。F3 把最轻的 stage 放到 PP3,瓶颈 274.6 → ~250 ms。

#### 显存账本(`restart_F3.log`)

| stage | 现切分 权重/卡 | F3 权重/卡 | 现切分余量 | F3 余量 |
|---|---|---|---|---|
| PP0 | 18.44 GB(10 层+embed) | **20.14 GB**(11 层+embed) | 0.93 GB | **0.64 GB** |
| PP3 | 20.15 GB(11 层+mtp+head) | **18.45 GB**(10 层+mtp+head) | 0.93→2.65 | **2.65 GB** |
| PP1 / PP2 | 19.15 / 19.17 GB | 19.15 / 19.17 GB | 1.58 / 1.63 | 1.57 / 1.62 |

**PP0↔PP3 精确对调 1706 MiB**(每层 TP2 分摊 3413/2),KV 池 153600 → 162304。

#### 吞吐 A/B(`ab_compare.py A F3`,同构建、同脚本、同载荷历史)

| lens | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| 1000 | 1.041 | 1.047 | 1.080 | **1.092** |
| 4000 | 1.077 | 1.083 | 1.090 | **1.100** |

- **prefill 8/8 格子全正,+4.1% ~ +10.0%,均值 +7.6%** —— 与文档记的 +8.2% 吻合;
- **decode 不动**:0.999 / 1.019 / 0.962 / 1.099,bs=1 对照 **0.999**(两臂热状态可比)。
  bs=2/4/8 的抖动 = 两臂自身 spread(12.8~15.3% vs 4.6~19.4%),不是 flag 的效应;
- K scaling 1.75× vs A 臂 1.67× —— 略好,方向一致。

#### 闸门

| 闸门 | 结果 |
|---|---|
| greedy parity(臂内 2 次) | **6/6 byte-identical** |
| **跨臂 byte-identical(A vs F3)** | **6/6** —— 层边界变了但每层输入张量不变,数学上**就该**一致,实测确实一致 → **+7.6% 是纯调度收益,没有偷换算法** |
| decode graph 覆盖 | 100.0%(774/774 行) |
| **200 题 GSM8K 并发 8** | **跑完 200/200,acc 0.940,服务存活** |
| 崩溃计数 | `unhandled cuda error` 0 / `out of memory` 0 / IMA 0 / `Scheduler hit an exception` 0 |

GSM8K 0.940 vs 旧基线 0.950:200 题下 1 个百分点 = 2 题,σ≈1.5% → 0.7σ,不是回归;
跨臂字节一致已经排除数值原因。

#### 结论

**F3 通过全部闸门,建议上线。零代码、纯环境变量 `SGLANG_PP_LAYER_PARTITION=11,11,11,10`,
prefill +7.6%。**

**但记一笔风险**:PP0 余量已到 **0.64 GB**,这是全局最紧的一档。以后任何给 PP0 加显存的
改动(加 draft、加 graph 桶、加大 ctx)都会先撞它 —— EAGLE 之所以能装下,正是因为它
落在 PP3 而不是 PP0。

---

### 6.10 EAGLE MTP 投机解码:**算术上不可能,结案**

**先说这次证否纠正了旧文档的什么。** `SM75_DSV4_DECODE_PROFILE.md` 记的 OOM 原因是:

> under PP the draft keeps its own copy of the input embedding on the last stage
> (129280x4096 fp16 = 1010 MiB per call) where 0.79 GiB is free after weights.
> Making that embedding shard-local and fp8 would fit, at the cost of ~60% of the KV pool.

**这个归因是错的。** 1010 MiB 的 embedding 只是 draft 的一小块;真正压死的是
**draft 模型本身的权重**。按 safetensors metadata 精确统计(checkpoint `*.safetensors` 逐 key 累加):

| 组件 | 总量 | **每卡 @TP2** |
|---|---|---|
| decoder 层(43 层) | 153.25 GiB | 76.62 GiB |
| **MTP / draft(nextn)** | **10.68 GiB** | **5.34 GiB** |
| embed_tokens | 0.99 GiB | 0.49 GiB |
| **合计** | **164.91 GiB** | **82.46 GiB** |

**8 卡 × 22 GB = 176 GB 总容量,权重 164.91 GB = 93.7%。** PP4×TP2 已是最省的切法,
再叠一份 10.68 GiB 的 draft 副本 → **越过物理上限**。

#### E1 臂实测

`SGLANG_PP_LAYER_PARTITION=11,11,11,10` + `SGLANG_ENABLE_PP_SPEC=1` +
`--speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-eagle-topk 1`:

```
ValueError: Loaded weights leave no GPU memory for the KV cache under
--mem-fraction-static=0.97. Raise --mem-fraction-static above 0.999
(minimum viable = 1 - available/pre = 0.9987).
```

- `available/pre = 0.0013` → **加载完只剩约 27 MB**(不是擦边,差好几个 GB);
- 崩在 **PP0 / PP1**,**不是 PP3** → **draft 权重被算进每张卡的预算,不是只落最后一个 stage**;
- E1 根本没走到 `Load weight end` —— 死在 `kv_cache_configurator._profile_available_bytes`。

#### 逐条排除"是不是还有别的旋钮"

| 想法 | 判定 |
|---|---|
| F3 把 PP3 余量抬到 2.65 GB | **无效** —— draft 不只落 PP3(见上) |
| 降 `--mem-fraction-static` | **无效且已到下限** —— 报错就是它抛的;可用区间只有 0.9548–0.97 |
| draft 不做 TP 切分 | **更糟** —— 要背整份 10.68 GiB,而不是 5.34 |
| 把 draft 权重也做 PP 切分 | **开关不存在** —— 查 `arg_groups/fields/spec.py` 与 `environ.py`,无此项 |
| `--speculative-draft-kv-cache-dtype fp8_e4m3` | **无关** —— 那是 draft 的 **KV pool**(运行时),5.34 GiB 是**加载时**的权重 |
| DSPark 替代 | 旧文档已否:非 `pp_size==1` 一律拒绝,且 8 卡放不下两个角色 |

#### 结论

**EAGLE / MTP 投机解码在这台机器上不可能。** 这不是调参问题,是 8×22 GB 装不下
164.91 GB 权重 + 10.68 GiB draft 副本。**不要再投入。**

`deepseek-v4-flash.sh` 里为它加的 `SPEC_ALGO` 门控**保留**(默认空 = 不追加任何参数,
默认命令行与改动前逐字节一致),留作将来换大显存机器时的入口。

---

## 7. 正确性闸门:贪心 parity —— **PP_EARLY_PROXY_SEND 通过**

`capture_greedy.py` 的前提是 "byte-identical greedy output across two builds"。完整实测矩阵:

| 样本 | 实例 | flag | 结果 |
|---|---|---|---|
| A0 | 实例 1(**跑过 pp_probe + pf_conc_probe**) | 0 | **异类**:prompt 2 → 214 char、prompt 5 → 206 char |
| A1, A2 | 实例 2 | 0 | 彼此一致:94/183/**198**/152/141/**217** |
| B1, B2, B3 | 实例 3 | 1 | 彼此一致,且 **与 A1 逐字节相同** |

**结论(三重验证):**

1. `SGLANG_PP_EARLY_PROXY_SEND=1` **对输出零影响** —— `A1 vs B1` 6/6 逐字节相同。
   最初那次 `PARITY FAIL: 2/6` 是 **A0 单独异常**造成的假阳性,不是 flag 的问题。
2. 同实例内确定性:A1==A2,B1==B2==B3。
3. **负载历史会改变贪心输出**:A0 所在的实例在采样前跑过 8 并发长 prompt 探针,
   唯独它分叉。合理机制是**批组成** —— `pp_max_micro_batch_size=2` 允许两条请求同批,
   批量 GEMM/MoE reduce 的归约顺序与单条不同 → 数值微差 → 分叉 token。
   (A1/A2/B* 都是**空闲实例上的串行采样**,批大小恒为 1。)

→ **使用规则**:
- 跨 arm 比较前,两个 arm 都必须在**相同的负载历史**下采样(都是冷启、都在探针之前)。
- **不要**拿一个跑过负载的实例的输出当基线。
- `greedy.py compare` 有个 bug:两行都是 error 时 `text=""` 会被判成
  "IDENTICAL (0 chars)" —— **必须先检查 `error` 字段再比较**。

---

## 8. 工具

| 文件 | 用途 |
|---|---|
| `ab_restart.sh` | 停/起/等健康 + 打印本次 arm 的显存账 + OOM 检查。launcher 用 `>"$log"` 截断日志,所以整份日志就是本次 arm |
| `dec_probe.py` | decode 吞吐,K 组并发交错取中位数;顺带统计 `cuda graph: True` 占比 |
| `pp_probe.py` | 8 并发窗口内原始调度日志 dump + `/get_server_info` 关键字段 |
| `pf_conc_probe.py` | prefill 并发扫描 + 窗口内 `#new-seq`/`#pending` 直方图 |
| `greedy.py` | greedy parity 捕获与比对(`capture_greedy.py` 硬编码 `:30000` 且无比对模式) |

踩过的坑:

- `ab_restart.sh` 第一版用日志 offset 定位 → launcher 是 `>"$log"` **截断**,offset 无意义。
- Decode 日志正则顺序写错过两次(`gen throughput` 在 `cuda graph` **之后**、`#queue-req` 在最末),
  导致直方图空 —— 直方图为空时先怀疑正则,不要当成"没有数据"。
- 用字节 offset seek **文本模式**文件会错位,要 `open(..., "rb")`。
- `ab_restart.sh` 被 SIGKILL 后,launcher 里 `setsid nohup sglang serve &` 的子进程会变成孤儿
  继续加载权重 → 下一轮 `check()` 报 "a GPU holds 20536 MiB"。
  **停机后必须同时确认 `nvidia-smi --query-compute-apps` 为空,而不只是 `pgrep` 为空。**

---

## 9. 本轮之后的优先级(按证据强度排序)

### A. 已证否 —— 不要再投入

| 项 | 硬证据 |
|---|---|
| D1 抬 graph 桶(原形式) | `#running-req ∈ {1,2}`,`cuda graph: True` 100%(1336/1336 与 680/680/720/720 行) |
| §4 缩 `--max-total-tokens` | `available_bytes=0.31 GB` → flag 根本没绑定;**这也解释了当年"缩池救不了 PP0"为什么没效果 —— 测错了旋钮** |
| `num_continuous_decode_steps` | 只在 `arg_groups/fields/schedule.py:194` 与 `field_order.py:67` 出现,**全仓无消费点** |
| chunk 512→1024 | `PREFILL_PROFILE.md:657` 已实测 1419/1410/1411 = 零收益 |
| `SGLANG_PP_EARLY_PROXY_SEND=1` | 三臂交错 −27~30%(§4) |
| F4 "AR spin 小 grid" | `k3_ar_fusion.gemm_ag_up_fits` 要求 **`state.world_size == 8`**(kimi_k3 专用),本机 **TP2** → 不适用 |
| headshared / rowgather / MoE FP8 acc / 128KB swizzle / L1 / wire 带宽 / 按 chunk 流水重叠 | 项目文档已证否 |

### B. 生产稳定性 —— **已定位并修复**(§6.5→§6.7)

**已确证的**:IMA 真实存在、可复现、发生在 greedy 前 1~6 个 prompt、受害者恒为 **PP2/PP3**。
**根因**:`dsv4/topk.py` 的 `_topk_transform_paged_triton_kernel` 缺 `page_id < page_table_width`
上界(`page_table_width` 是入参但**从未使用**)→ 脏 page id(12673630)→ `extra_indices` 被污染
→ 稀疏 attention 只判 `raw >= 0` 且拿不到 `num_pages` → 取址飞出 KV 池 → IMA。
**已被证否的**:① 服务期 Triton 懒加载;② prefill graph 顺带消 IMA;
③ `_headshared_sparse_kernel` 越界是**单独**首因(它只是共因的两个出口之一);
④ 错误在 decode CUDA graph replay 内;⑤ coredump/gdb 取证路线(工具链堵死)。
**方法论结论:异步 CUDA 错误的 Python 归因,只有在同步链路里才可信** ——
`CUDA_LAUNCH_BLOCKING` 只管 Runtime API(Triton 走 `cuLaunchKernel`),于是四份档案四个
互不相干的报告帧;`SGLANG_TRITON_SYNC_EVERY_LAUNCH=1`(launch 后立即 device-wide sync)
给出的栈才可信,归因上界是"**晚一个 launch**"。

| # | 动作 | 状态 | 为什么排这个位置 |
|---|---|---|---|
| 1 | `SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB=999` | **已带上,每臂都有** | 让"上卡时刻"可与栈对表;**也正因为记全了,才发现加载与崩溃并不同因** |
| 2 | `CUDA_LAUNCH_BLOCKING=1`(LBO1) | **✅ 完成,证否** | 只改**报出时机**(PG3 与 LBO1 同配置,一个有栈一个没有),不改**可信度**;它只管 Runtime API,Triton 走 `cuLaunchKernel` |
| 3 | PG4:`SGLANG_SM75_HEADSHARED_MIN_BATCH=0` | **✅ 完成,证否** | 0 次 headshared 上卡 → 开关生效;照样崩,帧换成 `dequant_block_fp8_slice` |
| 4 | LBO2:`GRAPH_BACKEND_DECODE=disabled` | **✅ 完成,证否** | 0 次 decode 捕获已确认;照样崩,帧换成 `moe_swiglu_clamp`,rank 从 PP2 漂到 PP3 |
| 5 | CD1:`SGLANG_CUDA_COREDUMP=1` + `cuda-gdb -c` | **✅ 完成,工具链堵死** | dump **成功且完整**(2×199,755,747 B,`All done`),符号/行表/寄存器俱全;但 `e_phnum=0` → **cuda-gdb 12.0 / 12.9 / 13.2 + 系统 gdb 全拒**;最小复现证明**与 sglang 的 `skip_*` flags 无关**;`log_only` 环境变量**不生效**(§6.7) |
| 6 | **SY1:`SGLANG_TRITON_SYNC_EVERY_LAUNCH=1`** | **✅ 完成,可信栈到手** | 栈 = **`_headshared_sparse_kernel`**(受害者 PP2 `rank5`);launch 后**立刻** device-wide sync → 归因上界"晚一个 launch",而 launch 就在 sync 正上方 → **可信**。⚠ 该臂吞吐数据作废 |
| 7 | **PG5:关 headshared + 同步钩子** | **✅ 完成,"第二个坑"证否** | headshared 上卡 **0 次**(开关生效)→ 栈换成 **`_tiled_sparse_decode_kernel`**,但**同样在 `flash_mla_sparse_decode_triton`** → **一个根因,不是两个**。PG4 的开关有效性另用加载计数复核(114 个上卡里 headshared 0 个 vs 其它臂 33–34) |
| 8 | **SV1→SV2:`SGLANG_VALIDATE_SPARSE_INDICES=1`** | **✅ 完成,根因坐实** | SV1 因 `.any()` 在 capture 期同步设备而**启动失败**;SV2 命中 **4 次** `SPARSE_INDICES_OOB [extra]`(全在 PP2、1/7168、`max=811112383` vs 池 38656),**夹紧后 IMA 消失、greedy 6/6 全过** |
| 9 | **修复:`topk.py` 补 `page_id < page_table_width`** | **✅ 完成,根因闭合** | 元凶 = `_topk_transform_paged_triton_kernel` 里 **`page_table_width` 传入但从未使用**;脏 page id 经 `extra_indices` 流进 attention(后者又缺 `num_pages` 上界)→ IMA。**判据达成:OOB 计数 4 → 0**(校验器是否真开用 `/proc/<pid>/environ` 核过,不是"没开所以没报") |
| 10 | **prefill graph 吞吐 A/B + greedy parity** | **✅ 完成,整类关闭** | A vs PG3 同构建交错测:**prefill 0.368×(慢 2.9 倍)**,8/8 格子一致、bs=1 对照 0.993;`full` 后端 **dsv4 未实现**、启动即崩;parity 臂内不自洽(prompt 0 近平局)。**生产保持 `disabled`**。见 §6.8 |
| 11 | ~~prefill CUDA graph 消 IMA~~ | **已证否**(§6.4) | breakable 断点体在每次 replay 都跑 eager,服务期上卡照旧 |
| 12 | ~~`SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY=1`~~ | **已作废** | 它只能给一个 kernel 名,而"kernel 名"正是我们已经污染过的那个量 |

> ⚠ **`SGLANG_SM75_HEADSHARED_MIN_BATCH=0` 已证明消不掉 IMA**,别再拿它当"稳定性修复"。
> 它仍可作为**性能 A/B 开关**(`environ.py:1148-1161`:head-shared 在 86K ctx bs=1 是
> **13.55 vs 11.00 tok/s = +23%**,"wins at every batch size")。
> **PG4 的 decode 数与 A / M16 不可比**,只有 prefill 数可比。

**已从 IMA 缓解方案中移除:F3 层重排放 1706 MiB。**
理由(§6.2):40 次 `cuModuleLoadData` **全部成功**(0.79 → 0.61 GiB 只掉 180 MB),
而且 triton_load_watch 讲的是 **CUDA OOM**,现场是 **IMA** —— 放显存只能消掉告警行,
**不构成修复**。F3 的价值应只按 §9-C.4(+8.2%)与解锁 EAGLE 来算,不要算进稳定性。

> **这条旧观察现在要重新定性**(§6.5):`scheduler.py:1861-1863` 说"任何 serving 期
> device-load 都是懒加载",没错 —— 但它**与 IMA 无关**。真首因是
> `_headshared_sparse_kernel` **被调用**时越界,加载时刻只是巧合地同秒出现。
> **不要再把精力花在"预热覆盖"上**,那条线索已作废。

### C. 吞吐 open 项

| # | 项 | 预期 | 成本 | 状态 |
|---|---|---|---|---|
| 1 | **D1'**(`MAXREQ` + graph 桶同时抬) | **实测 +5.8% decode(仅 N≥16);N=8 反而 −4.9%** | 纯配置,最便宜 | **已关闭**:模型 +52% 被证否,bs 再翻倍预计 ≤+4%,**不再跑 32**;§3 末有完整数据与推论 |
| 2 | ~~prefill all-reduce 效率 +6%~~ | **机制已被 `PROGRESS.md` #4 证否,不是 open 项** | — | 见 §5.1:0.216 ms 里绝大部分是**等 peer 的 skew**,不是传输效率 |
| 3 | MoE k-split(align/partial buffer 按 live rows 定尺寸) | **+3–5% decode** | 中等代码 | **本轮重新核过,文档与源码矛盾**:fused reduce **已在建好的 .so 里**(`nm -D` 见 `__tvm_ffi_reduce`,`W4A16PtxV3Reduce::run`、`w4a16_ks_reduce_kernel` 都在)→ `torch.sum` 兜底走不到;`environ.py` 实测 KS2 = **0.087 ms(1.33×)**,而文档同段引 0.116(1.00×)—— **两代数据混写,1.00× 那组是 fix 之前**。生产 bs=1 已在 NT4+KS2 最优点;**真正没测过的是 bs=1 的 KS4(cfg 8)**,但 `fp8.py` 只暴露 cfg 7/9,需改一行代码。**优先级降到"便宜但小"** |
| 4 | **F3 层重排 `11,11,11,10`** | **实测 prefill +4.1~+10.0%(均值 +7.6%)** | 纯配置,零代码 | **✅ 通过全部闸门,建议上线** —— 见 §6.9。**旧文档的"不可用"结论已过期**:9/29 记的是 PP0 余量 0.42 GB 崩,今天实测 **0.64 GB** 且 200 题 GSM8K 零崩溃 |
| 5 | ~~**EAGLE MTP 投机解码**~~ | — | 配置 | **❌ 算术上不可能,结案**(§6.10)。**旧文档的归因是错的**:它说 OOM 是"draft 复制了一份 1010 MiB 的 embedding",实测 draft 权重整份 **5.34 GiB/卡 @TP2**,复制到每张卡 → 直接越过物理上限。F3 抬 PP3 余量**救不了**,因为 draft 不只落在 PP3 |
| 6 | hc_pre cublas chain / 46 个 FillFunctor | 各 ~1% | 小 | 文档列的 1、2 |
| 7 | ~~**修 `_headshared_sparse_kernel` 越界 = 修 IMA 首因**~~ | **方向对、对象错**:headshared 确实会越界,但只是**共因的两个出口之一**(§6.7) | — | SY1(开)/ PG5(关)两份可信栈分别落在 `_headshared_sparse_kernel` 与 `_tiled_sparse_decode_kernel`,**同在 `flash_mla_sparse_decode_triton`**;真修在 `topk.py` 的 `page_id < page_table_width` 上界。headshared 在 86K ctx bs=1 是 13.55 vs 11.00 tok/s = **+23%**,**关不得** |

**C.4 成立、C.5 结案**:F3 把 1.7 GB 余量从 PP0 搬到 PP3。**PP0 变紧(0.93 → 0.64 GB)但够用**
(200 题 GSM8K 零崩溃);PP3 变富,但 **EAGLE 根本不落在 PP3** —— draft 权重每卡都要背
(§6.10),所以这份富余对 EAGLE 无用。

⚠ **F3 上线后 PP0 余量只剩 0.64 GB,是全局最紧的一档。** 以后任何给 PP0 加显存的改动
(加 draft、加 graph 桶、加大 ctx、调高 mem-fraction)都会先撞它。§2 的发现仍然成立:
**缩 KV 池救不了 PP0**(池已缩到底)。
正解是 PP0 自身的 `available_bytes` 会随权重自动收缩,真正要问的是
**PP0 那 0.42 GB 之外还能不能挤出 0.3–0.5 GB**(embed_tokens 每卡 505 MiB 是最大单项)。
