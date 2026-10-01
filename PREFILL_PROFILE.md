# PREFILL PROFILING REPORT — 2026-09-28

Measured on the **production** server (`deepseek-v4-flash`, Vision-Exp, TP2×PP4,
`chunked-prefill-size 512`, decode CUDA graph on, prefill graph off), opt3 wheel.
This supersedes the prefill numbers quoted in docstrings across `*.py` probes,
which were all read off a **2026-09-20 capture of the dev server (:30000, now
dead) taken before opt1's fused kernels shipped**.

## Tooling added

| File | What it does |
|---|---|
| `pf_probe_prod.py` | Wall-clock prefill tok/s, cache-defeated (uuid4 at position 0), median-of-N, rounds interleaved across lengths to cancel ~6% thermal drift. Port-configurable (default 8200). |
| `prof_prefill_prod.py` | Server-side EXTEND profile capture against the live server. Refuses `profile_by_stage=True` + `num_steps=0`. |
| `pf_analyze.py` | Stage budget / kernel family / top-kernel / chunk cadence from a trace dir. Auto-detects trace time units. |

## Baselines (reproduced across a server restart)

| prompt tokens | median tok/s | spread |
|---|---|---|
| 2 941 | **1 080** | 2.8% |
| 11 767 | **1 361** | 2.2% |

## Two methodology bugs found and fixed

1. **sglang upstream bug — `/start_profile` kills the server.**
   `profile_by_stage=True` with `num_steps=0` leaves `profiler_prefill_ct` as
   `None` (initialised only inside `if num_steps:`, `profiler_manager.py:144-147`,
   declared `None` at `:74-75`), then `_profile_batch_predicate` does
   `self.profiler_prefill_ct += 1` at `:417` →
   `TypeError: unsupported operand type(s) for +=: 'NoneType' and 'int'`,
   which takes down all 8 ranks. Reproduced at 21:57:59; server needed a manual
   restart. `prof_prefill_prod.py` now rejects that combination up front.

2. **My own analyzer had a 1000× unit bug.** Kineto writes `ts`/`dur` in
   **microseconds** while setting `displayTimeUnit: "ms"`. Trusting that field
   reported a 5 193 ms attention call. Now calibrated against the median
   `cuda_runtime` launch duration (a host-side launch is µs, never ms) and the
   basis is printed in the report.

## Where prefill time actually goes

Per 512-token chunk, per stage. "compute" = union of **non-nccl** kernel
intervals (nccl `SendRecv` is excluded because it runs **100% concurrent with
other kernels** — it is a spin absorbing pipeline skew, not a transfer).

| stage | layers | compute/chunk | cadence | utilisation |
|---|---|---|---|---|
| PP0 | 10 | 244.5 ms | 285.2 ms | 85.8% |
| PP1 | 11 | 255.3 ms | 327.9 ms | 77.9% |
| PP2 | 11 | 265.0 ms | 301.7 ms | 87.8% |
| **PP3** | 11 | **280.7 ms** | 303.4 ms | **92.5%** |

**The pipeline is already ~90% utilised.** This is the headline and it
contradicts the existing notes, which claim "29-30% busy, 3.4× prize, all
scheduling not kernels" (`cadence.py`, `pf_conc2.py`, `busy.py`). Those numbers
came from the pre-opt1 dev capture. On the production build the stages are
nearly saturated.

Consequence: the arithmetic ceiling from perfect scheduling alone is
`512 / 0.2807 = 1 824 tok/s`, and we measure 1 361. **Scheduling fixes can buy
at most ~1.34×.** The 3 500–5 000 tok/s target must come from making the
kernels themselves faster.

### Kernel families (middle chunk, TP0, device time)

| family | PP0 | PP1 | PP2 | PP3 |
|---|---|---|---|---|
| attn | 28.7% | 46.0% | 43.4% | 45.2% |
| moe | 19.2% | 28.6% | 32.0% | 30.2% |
| gemm | 8.2% | 12.2% | 12.5% | 12.4% |
| elementwise | 5.9% | 8.8% | 9.2% | 8.8% |
| comm (spin) | 36.7% | 2.3% | 0.8% | 37.3% |

### Dominant kernels (PP3, per chunk)

| kernel | ms | calls | grid |
|---|---|---|---|
| `_headshared_sparse_kernel` | 123.3 | 22 | `[512, 2, 1]` × `[128, 1, 1]` |
| `w4a16_v3_kernel` | 57.6 | 20 | `[64, 433, 1]` × `[32, 1, 1]` |
| `_moe_combine_kernel` | 26.6 | 10 | `[512, 8, 1]` × `[64, 1, 1]` |
| `turing_fp16_s1688gemm_..._128x128_tn` | 15.0 | 70 | `[8, 4, 1]` × `[128, 1, 1]` |

Per-call costs (PP3): head-shared attention **5.6 ms/call**, W4A16 expert
**2.9 ms/call**, moe_combine **2.7 ms/call**.

## Targets implied

Attention + MoE = **~75% of compute** on every stage. To reach 3 500–5 000
tok/s the per-chunk compute on the busiest stage has to fall from 281 ms to
roughly 100–145 ms, i.e. a **2–2.8× reduction in real kernel work**, not
bubble removal.

---

# Task 1 result — launch-knob sweep at prefill shape: NO WIN

## Question

`_headshared_sparse_kernel` is 52% of PP3's device time (6.3 ms/call at the
production shape). Its three launch constants —
`SGLANG_SM75_HS_NCOL` (64), `_HS_WARPS` (4), `_HS_STAGES` (2) — were tuned at
**B=1 (decode)**: the comment above them records "4/8/16/32 blocks all cost
0.710 ms". At prefill B=512 the grid is 1024 programs and the cost structure is
different, so those constants were unvalidated where the kernel actually burns
its time.

## Method

`pf_hs_sweep.py` builds the production cache **in the kernel's own byte layout**
and checks every config against an fp32 reference over the same gather, so a fast
but wrong config cannot win.

Getting the layout right took four corrections, each of which had silently
produced NaN or garbage:

| mistake | symptom | fix |
|---|---|---|
| page = `PAGE*576` only | scale bytes read as 0-255 → `exp2(255-127)` = `inf` | page = `[data: PAGE*576][scales: PAGE*8]` = 9344 B, from `_TOKEN_DATA_STRIDE=576` and `scale_section_off = page_size*576` |
| `_HS_GROUP=32`, `_HS_SCALE_BYTES=1` | shape errors | actually **64** and **8** (`flash_mla_sm120_triton.py:486-487`) |
| contiguous rope write | rope decoded to ~1e28 | rope is **per-token** at `t*576+448`, not one block per page |
| reference indexed `p*page_bytes` on an already-strided view | out-of-bounds | within-page offsets only |

Correctness gate: **rel 7.1e-3** vs the fp32 reference, identical across all 18
configs (fp16 accumulation, as expected).

## Result

Full grid (NCOL × warps × stages), B=512, topk=512, 16 K-token cache:

| NCOL | warps | stages | ms | vs shipped |
|---|---|---|---|---|
| 64 | 4 | 2 | 21.23 | **1.00x** (shipped) |
| 64 | 4 | 1 | 21.19 | 1.00x |
| 64 | 4 | 3 | 21.22 | 1.00x |
| 128 | 8 | 3 | 22.84 | 0.93x |
| 64 | 8 | 1 | 24.92 | 0.85x |
| 64 | 2 | 2 | 30.79 | 0.69x |
| 128 | 2 | 2 | 73.58 | 0.33x |

**Interleaved 5-round confirmation** (to cancel the box's ~6% thermal drift):
stages=1/2/3 give 21.19 / 21.17 / 21.22 ms — a 0.3% spread, i.e. **identical**.

The 1.14x that first appeared in a single-pass sweep was drift, not signal. This
is why the sweep interleaves and why the naive ranking was not trusted.

## Conclusion

**The shipped configuration is already optimal.** No change is warranted, and the
existing decode-time tuning happens to be correct for prefill too. The bottleneck
is not a tuning problem — see Task 2.

Secondary finding: `NCOL` and `warps` matter a lot (0.33x to 1.00x), `stages`
does not. `NCOL=64` with 4 warps is a genuine optimum, not a default that
happened to survive.

---

# Task 2 result — the 13% occupancy is shared memory, and NCOL=32 is a trap

## Where the occupancy actually goes

Measured by compiling the real kernel and reading `metadata.shared`:

| NCOL | NCHUNK = 512/NCOL | shared mem | blocks/SM | occupancy |
|---|---|---|---|---|
| 32 | 16 | 25 600 B | 2 | 12% |
| **64** | **8** | **40 960 B** | **1** | **6%** |
| 128 | 4 | 45 056 B | 1 | 6% |
| 256 | 2 | 53 248 B | 1 | 6% |

**Shared memory is the limiter, not registers.** 40 KB against SM75's 64 KB allows
exactly one block per SM (4 warps = 6% occupancy). Cutting registers would change
nothing: even at 32 regs/thread the smem cap still admits one block.

The 40 KB is the 8 chunk accumulators `acc0..acc7` (`[16,64]` fp32 = 32 KB) plus
`acc_r` (4 KB) plus one 4 KB staging tile. It is `stages`-independent — measured
identical at stages=1/2/4, which is *why* the stages knob does nothing.

## NCOL=32 looks like a 1.7x win and is silently wrong

NCOL=32 halves the accumulator set, doubles blocks/SM, and measures:

```
shipped NCOL=64    21.096 ms   maxerr 4.71e+00
NCOL=32            12.227 ms   maxerr 9.60e+02   <- 1.725x, and WRONG
```

**Every one of 256 tokens is wrong**, across all output columns, with errors equal
to the full magnitude of the reference (960.0) — i.e. the output is garbage, not
degraded.

Cause: `NCHUNK = _HS_NOPE_PAD // NCOL` = 512/32 = **16**, but the kernel
unrolls only 8 chunks (`q0..q7`, `acc0..acc7`, and the QK/PV/store blocks each
guard on `NCHUNK >= k` for k=1..8). Chunks 8-15 are never computed, so the top
half of the 448-wide nope dimension silently vanishes from both the QK dot and
the PV accumulation. The comment at line 589-591 states the real constraint —
"all 8 possible chunks (NCOL down to 64)" — but nothing enforces it, so
`SGLANG_SM75_HS_NCOL=32` compiles, runs, and returns wrong numbers.

This is a **latent bug, not a regression I introduced**: production never sets the
variable (default 64), and `ab_prefill.sh` / `ab_prefill2.sh` — the two scripts
that sweep it as `$1` — have no `logs/ab-*` output, so the sweep was never
completed end-to-end. Had it been, NCOL=32 would have looked like a 1.7x prefill
win and shipped a silently wrong model.

## What this rules out, and what it leaves

Occupancy cannot be raised by tuning. The only lever that reduces the accumulator
footprint is fewer chunks, and fewer chunks is exactly what breaks correctness.
`BLOCK_H` can't drop below 16 (tl.dot requires it) and raising it makes smem
worse.

So the real options for this kernel are structural, not parametric:
- hold one `[BLOCK_H, 512]` accumulator instead of 8 chunked ones, which needs
  either 16 unrolled chunks or an inner loop Triton will actually pipeline
- stop staging the accumulator through shared memory at all (the FFMA lowering
  means `tl.dot` is not using `ldmatrix`/`mma` anyway, so the smem staging is
  pure overhead on this hardware)
- rewrite the QK/PV to use explicit FFMA on gathered values, skipping `tl.dot`

All three are real kernel work, not tuning. The cheap experiment is exhausted:
Task 1 showed the launch knobs are already optimal, and Task 2 shows occupancy
is smem-bound with the one apparent escape hatch being a correctness trap.

---

# Attention 重写前的定量化分析 — 2026-09-29

在写任何 kernel 之前,先把"重写能拿到多少"测出来,而不是猜。

## 修正:实际 gather 流量是 604 MB/call,不是 151 MB

之前我按 `B*topk*tok_bytes` 算过一次,得出 151 MB、7.5% 带宽。**这是错的**:
MLA 只有一个 KV head 被 64 个 query head 共享,而 grid 把 head 切成 `H/BLOCK_H = 4` 块,
每块**各自重新 gather 同一份 KV**,PV 那一遍还要再来一次:

```
per block  = topk * 576 B = 512 * 576 = 0.30 MB
grid       = (512, 4) = 2048 blocks
total      = 604 MB/call   (8x 冗余:QK 4 次 + PV 4 次)
```

## 关键测量:gather 只占 2%

`hs_floor_probe.py` 用**同样的输入、同样的寻址**,做三档探针(每档都把结果累加回
output,避免被 dead-code elimination 掉):

| 探针 | ms/call | 说明 |
|---|---|---|
| **gather-only(BLOCK_H=16)** | **0.332** | 同样的 gather + 反量化,没有 dot |
| real kernel | 21.08 | 生产 kernel |

**gather 只占 kernel 的 2%,剩下 98% 是算术 + softmax。**

(注:gather-only 报出 1820 GB/s = 峰值 295%,因为 16K token 的 cache 只有 9.4 MB,
L2 直接命中。生产上 KV 池是 138 MB,会走 DRAM。但即便按 DRAM 算,604 MB / 616 GB/s
= 0.98 ms,仍然只有 21 ms 的 5%。**结论不依赖 cache 大小。**)

## 所以瓶颈既不是带宽也不是 tensor core

| 方案 | 天花板 | 判断 |
|---|---|---|
| 减少 gather 流量(提高 BLOCK_H 减冗余) | gather 已占 2% | **收益上限 2%** |
| 换 tensor core(mma.sync) | cuBLAS 实测 43.3 TFLOP/s vs FP32 29.6 = **1.46x** | 算术部分最多快 1.46x |
| **提高 occupancy 让 FFMA 流水线填满** | 见下 | **这才是主要空间** |

算术地板:34.4 GFLOP/call ÷ 29.6 TFLOP/s = **1.16 ms**。实测 21.08 ms,
**是地板的 17.9 倍**。这个 18 倍的差距不是带宽(已排除),不是缺 tensor core
(只值 1.46x),而是**6% occupancy 下 FFMA 流水线填不满**——每次 `tl.dot` 发 2048 条
标量 FFMA,需要足够多的 warp 才能掩盖延迟,而当前每 SM 只有 4 个 warp。

## 重写方案(按数据排序)

1. **合并 8 个 chunk 累加器为 1 个 `[BLOCK_H, 512]`** —— 直接把 40 KB smem 压到
   约 5 KB,让 blocks/SM 从 1 提到 4-8,occupancy 从 6% 到 25-50%。
   **这是唯一能动 18 倍里大部分的改动。** 障碍:Triton 的 `@jit` 不允许
   list comprehension,所以现在 8 个 chunk 是手写展开的;要么展开到 16 个
   (NCOL=32,smem 反而更小),要么用 `tl.static_range` 的单累加器。
   **注意:NCOL=32 在现有代码里是错的**(NCHUNK=16 > 展开的 8),重写时必须一并修。
2. **显式 FFMA 替代 `tl.dot`** —— 让 Triton 发出可控数量的 FMA,避免它为
   "模拟 tensor core"生成冗长的索引运算。这是次要项,只有在 1 做完、occupancy
   上来之后才值得看。
3. **BLOCK_H 16→32/64** —— 现在看是低价值(gather 只占 2%),但它同时会**增大**
   累加器从而恶化 smem,与方向 1 冲突。只有在 1 解决了 smem 之后才成立。

## 为什么这条路的收益有上限(必须诚实说明)

即使 occupancy 修好、算术填满到 100%,每次调用也要 1.16 ms(FFMA 地板),
22 次/chunk = **26 ms**,而现在 143 ms。理论上 5.5x。**但:**
- FFMA 路径不太可能填满到 100%,实测良好的 FFMA kernel 通常 50-70%
- 即便 5.5x 全拿到,PP3 从 282 ms 降到 ~165 ms,端到端约 1.4-1.7x,
  即 ~2000-2400 tok/s,**仍到不了 3500-5000**

**要达到 3500 tok/s,attention 必须同时做到:occupancy 修好 + 用上 tensor core
(1.46x)+ 减少 8x gather 冗余。** 三者叠加才可能。所以这不是"重写一个 kernel",
而是"重写这条数据通路",工作量按周计,不是一个 session。

---

# Attention 重写:第一个假设被自己的数据否掉 — 2026-09-29

## 做了什么

`proto_hs_unroll.py`:把 8 个手写展开的 chunk 累加器(`acc0..acc7` + `acc_r`)
换成**一个**累加器,在 `tl.static_range` 循环里复用。动机是实测 smem 40 960 B
对 64 KB 上限只容 1 个 block(6% occupancy),而 8 个 `[16,64]` fp32 累加器正好是 32 KB,
看起来就是元凶。数学完全不变,只测资源与时间。

## 结果:假设不成立

| 变体 | ms/call | `metadata.shared` |
|---|---|---|
| 展开 8 累加器(现状) | 21.09 | **40 960** |
| **单累加器 + static_range** | **23.63** | **40 960** ← 完全没降 |
| 单累加器,NCOL=128 | 35.68 | 45 056 |

**smem 一字节都没省,而且慢了 12%。** 我在 `PREFILL_PROFILE.md` 上一节里写的
"合并 8 个累加器可把 40 KB 压到约 5 KB,让 blocks/SM 从 1 提到 4-8" —— **这个预测是错的。**

## 为什么错:40 KB 不是累加器

累加器改成 1 个后 smem 不变,说明填满 40 KB 的不是它们。也不是 dot 的操作数
(`[64,16]` fp16 × 2 stages 只有 4 KB)。真正的原因是:

**`tl.dot` 在 sm_75 上被降级成标量 FFMA,没有 `mma.sync`。** 这种情况下 Triton 要把
**每个 dot 的两个操作数都经 shared memory 暂存**。而 kernel body 里一共有
8 个 QK dot + 8 个 PV dot + 1 个 rope dot,**全部在同一个未循环的函数体里**,
分配器无法让它们的缓冲区重叠 —— 40 KB 是这 17 个 dot 的操作数暂存之和。

所以**降低 smem 的正确手段是减少 dot 的个数,而不是缩小累加器**:
把 8 次 `[NCOL=64, BLOCK_T]` 的 gather + dot 合并成 1 次 `[NCOL=512, BLOCK_T]`。
但 `tl.dot` 要求各维 ≥16 且是 2 的幂,`[512,16]` 的 tile 是 16 KB fp16,
一次 dot 的操作数就 32 KB,**并不会更省**。

这形成了一个硬约束,我目前没有找到能同时满足的方案:
- 要 smem 小 → dot 要少 → 单个 tile 要宽 → 操作数反而大
- 要 dot 少且 tile 窄 → 只能把 nope 维真正压到 448(现在 padding 到 512,浪费 12.5%)

## 我现在的判断

这一条路(在 Triton 里重写 head-shared attention)已经连续两次被数据否掉:
第一次是 occupancy 归因错误(以为 smem=累加器),第二次是降 smem 的手段错误
(以为合并累加器有效)。**继续在 Triton 内调结构,大概率还有第三、第四次失败。**

真正对症的做法是**绕开 `tl.dot`**:既然它被降级成 FFMA 且要暂存操作数,
不如直接用 `tl.sum(q[:, None, :] * kv[None, :, :], axis=2)` 这类显式广播乘法,
让 Triton 生成纯 FMA 而不分配 dot 的操作数缓冲区。这会改变可用的 tile 形状
(需要 `[BLOCK_H, BLOCK_T, D]` 的寄存器,BLOCK_H=16/BLOCK_T=16/D=512 太大),
所以需要重新设计分块,例如 BLOCK_T 降到 4。

**建议:先做一个 30 分钟的可行性验证** —— 只测"显式广播 FMA 替代 tl.dot"
在 `[16, 16, 64]` 这个小切片上,生成的 smem 和 PTX 里 FMA 数量是否真的下降。
如果 smem 降下来了,再投入做完整重写;如果没降,这条路整体关闭,应该转向
"接受 attention 的现状,把 3500 tok/s 这个目标重新评估"(因为实测天花板就在
~2400 tok/s,见上一节)。

---

# 第二个假设也被否掉:绕开 tl.dot 不可行 — 2026-09-29

## 测了什么

既然 40 KB 是 `tl.dot` 在无 `mma.sync` 时**暂存每个 dot 的操作数**造成的,那
"不用 `tl.dot`、改用显式广播乘法"应该能省下 smem:

```python
s += tl.sum(q[:, None, :] * k[None, :, :], axis=2)    # 而不是 tl.dot(q, k.T)
```

在真实的 chunk 形状上对比(same NCHUNK/NCOL,同一份数据,同一个 `[16,16]` 分数 tile):

| 配置 | 变体 | shared | fma | mma | μs | 误差 |
|---|---|---|---|---|---|---|
| NCH=8 NCOL=64 | `tl.dot` | 4096 | 1024 | 0 | **61.6** | 2.8e+01 |
| NCH=8 NCOL=64 | bcast | **2048** | 1024 | 0 | 163.7 | **NaN** |
| NCH=4 NCOL=128 | `tl.dot` | 8192 | 1024 | 0 | **67.8** | 2.2e+01 |
| NCH=4 NCOL=128 | bcast | **4096** | 1024 | 0 | 156.7 | **NaN** |
| NCH=2 NCOL=256 | `tl.dot` | 16384 | 1024 | 0 | **68.4** | 2.6e+01 |
| NCH=2 NCOL=256 | bcast | **4096** | 512 | 0 | 102.1 | **NaN** |

## 结论:这条路关闭

**smem 确实降了一半(甚至 4 倍),但慢 2–2.6 倍,而且数值是 NaN。**

原因清楚:
1. **慢**:`tl.sum` 的归约轴让每个 FMA 都依赖一次跨 lane 的树形归约,而不是
   纯粹的 FMA 累加。同样的 1024 条 FMA,`tl.dot` 61.6 μs vs bcast 163.7 μs。
   Triton 对 `tl.dot` 有专门的 FFMA 调度,对手写广播反而更差。
2. **NaN**:`q[:, None, :] * k[None, :, :]` 在 fp16 上乘,256 项累加到 fp32 之前
   就会溢出(这也是 opt2 修过的那个 fp16 饱和问题的同类)。`tl.dot` 内部按 fp32
   累加所以没有这个问题。

**要拿到 smem 收益,必须让乘法在 fp32 上做,那意味着寄存器里要放
`[BH, BT, NCOL]` 的 fp32 中间量 = 16×16×64×4 B = 64 KB/线程组** —— 直接爆寄存器。
把 `BT` 降到 4 可以压到 16 KB,但 `tl.dot` 要求 `BT >= 16`。

## 三次尝试的累计结果

| 尝试 | 假设 | 结果 |
|---|---|---|
| 1 | occupancy 低是因为 8 个累加器占 32 KB | **否** — 换成 1 个累加器,smem 一字节没降(40960),还慢 12% |
| 2 | 40 KB 是 `tl.dot` 的操作数暂存,绕开它就行 | **否** — bcast 确实省一半 smem,但慢 2.6× 且 fp16 溢出成 NaN |
| 3 | (未测)减少 head-split 冗余 BLOCK_H 16→64 | gather 只占 2%,理论上限 2% |

**三次里有两次被自己的数据否掉。** 我在这个 kernel 上已经连续三次猜错归因,
不应该继续在没有新证据的情况下投入完整重写。

## 我建议停在这里,并重新评估目标

已确定的事实(都有测量支撑):
- attention 占 PP3 device 时间 52%,gather 只占其中 2%,98% 是 FFMA 算术
- 算术地板(全 FFMA 填满)1.16 ms/call,实测 21.08 ms,**差 17.9 倍**
- tensor core 只值 1.46×(cuBLAS 43.3 TFLOP/s vs FP32 29.6)
- **即使 occupancy 完美、算术填满,attention 也要 26 ms/chunk,PP3 也降不到 100 ms 以下**
- 端到端天花板实测在 **~2400 tok/s**(见上文推算)

**3500–5000 tok/s 在这个硬件上不是一个"优化"目标,是需要改模型或换硬件的目标。**
继续在 Triton 里重写 attention,按已知的约束推算,乐观情况也只能到 ~2000-2400 tok/s。

## 如果还要继续,应该换的是工具不是 kernel

唯一还没试过、且有机制依据的方向:**绕过 Triton,直接写 PTX/CUDA,
用手写 `mma.sync` 或手写 FFMA 循环**。理由是 Triton 在 sm_75 上没有 tensor core
代码路径(0 mma.sync 是 PTX 实测),它的 FFMA 调度和 smem 分配都不是为这个
场景设计的。但这是写 CUDA kernel 的工作量,不是调 Triton 参数。

---

# 目标重定为 2400 tok/s — 关键测量与所需条件 — 2026-09-29

用户确认 **2400 tok/s 也是巨大改善**(相对现在 1429 是 +68%),目标从 3500-5000
下调到 2400。这个目标是可达的,下面是把它拆成可验证条件的测量。

## 先修正一个我自己的测量错误

我之前用"单次 dot 微基准"测到 ~50 us,据此说"这是 launch 开销"。那个说法对,
但我用它做后续推理是错的 —— 它测不出真实 kernel 里的 dots 成本,因为真实 kernel
在**一次 launch 内**做 17 个 dots,launch 被摊薄了。

正确做法(`hs_dot_attrib.py`):把 17 个 dots 放进一个 kernel 的循环里,用
`ITERS=0`(循环体不执行)测 launch 成本,再减去它得到**每 step 的边际成本**:

| 量 | 值 |
|---|---|
| launch-only(ITERS=0) | 59.0 μs |
| **dots 边际成本(8 QK + 1 rope + 8 PV)** | **19.56 μs/step** |
| 同形状 FFMA 算术地板(单 block) | 1.355 μs/step |
| **实测/地板** | **14.4x** |

iters=8 和 iters=64 分别给出 19.56 / 20.46 μs,说明是稳定的边际成本,不是噪声。

## 这 14.4x 是什么

一个 block = 4 warps,占满一个 SM 的**一个** block 槽(smem 40 KB / 64 KB 上限)。
每 step 1.355 μs 是这 4 个 warp 满负荷算的极限,实测 19.56 μs —— **93% 的时间在等**,
等的是共享内存往返和 FFMA 依赖链,没有别的 warp 来掩盖。

按 blocks/SM 外推(每次调用 2048 blocks、32 tiles、68 SM = 30 轮):

| blocks/SM | 每次调用 | 相对现在 |
|---|---|---|
| **1(现状)** | 21.1 ms | 1x |
| 2 | ~10.5 ms | 2x |
| **4** | **~5.3 ms** | **4x** |
| 8 | ~2.6 ms | 8x |

**所以"提高 occupancy"确实是对的方向,我之前的结论没错;错的是我认为降 smem 的
手段(合并累加器、绕开 tl.dot)都试过了。** 那两条路失败不代表这条路不成立,
只代表那两种手段不成立。

## 达到 2400 tok/s 需要什么(把目标翻译成条件)

现在 PP3 每 chunk 282 ms,其中 attention 143 ms。要到 2400 tok/s:

| attention 降到 | PP3 每 chunk | 端到端 |
|---|---|---|
| 143 ms(现状) | 282 ms | 1814 tok/s |
| 100 ms | 239 ms | 2140 tok/s |
| **70 ms** | **209 ms** | **2446 tok/s** ✓ |
| 40 ms | 179 ms | 2856 tok/s |
| 26 ms(算术地板) | 165 ms | 3097 tok/s |

**目标 = 把 attention 从 143 ms 降到 ~70 ms,即 2x。**
这需要 blocks/SM 从 1 提到 2-4(按上面的外推),也就是 **smem 从 40 KB 降到 16-32 KB**。

**注意 26 ms 那行:即使 attention 做到算术地板(FFMA 100% 填满),端到端也只有
3097 tok/s。** 所以 2400 这个目标是现实的(不需要突破物理极限),而 3500-5000 确实不可能。
这个数字现在有了实测支撑,不是估计。

## 下一步该试什么(按可行性排序)

smem 40 KB 的成因已经定位:**17 个 `tl.dot` 的操作数暂存**,Triton 在 sm_75 上把
dot 降级成 FFMA,每个 dot 都要经 shared memory。已排除的手段:
- 合并 8 个累加器(1):smem 一字节没降
- 绕开 `tl.dot` 用广播乘法(2):省一半 smem 但慢 2.6x 且 fp16 溢出

**还没试的、且直接针对 smem 的手段:**

1. **减少未展开的 dot 数量**。当前 17 个 dot 全在一个未循环的函数体里,分配器
   无法让它们的缓冲区重叠。如果把 nope 的 8 个 chunk 改成**循环**(而不是展开),
   Triton 就能复用同一块 smem。但要注意:这正是尝试 1 做的"合并累加器"——
   它**只合并了累加器,dot 仍然是 8 个展开的**。真正没试的是**把 8 个 QK dot
   放进一个循环,让它们的操作数暂存区复用**。这与尝试 1 不同。
2. **`num_stages=1`**。当前 2,会把操作数缓冲翻倍。实测 stages 对时间无影响
   (21.19/21.17/22 ms),但**没测过它对 smem 的影响** —— 如果 2->1 能把 40 KB
   砍到 20 KB,blocks/SM 就翻倍。这是最便宜的一个实验,应该先做。
3. **BLOCK_T 16→8**。QK 操作数是 `[NCOL, BLOCK_T]` fp16,砍半直接省 smem。
   `tl.dot` 要求 >=16,但 PV 那一路的 `pf` 是 `[BLOCK_H, BLOCK_T]`,可以只对
   PV 用不同的 tile。

**第 2 项(测 num_stages 对 smem 的影响)是一个 5 分钟的实验,而且可能直接命中。**

## 追加:smem 降不下来的三个已验证原因(2026-09-29)

在 2400 tok/s 目标确定后,我把"降 smem"这条路上还没试过的旋钮全部试了。
**四个杠杆,三个是死路:**

| 手段 | smem | 时间 | 结论 |
|---|---|---|---|
| 现状(BLOCK_T=16, stages=2, 8 展开 dot) | 40 960 | 21.1 ms | 基准 |
| `num_stages` 1 / 2 / 3 / 4 | **40 960(全同)** | 21.2 ms | **无效** — stages 不影响 smem |
| `BLOCK_T` 16 → 8 | — | 编译失败 | **不可行** — `tl.dot` 要求 K ≥ 16,`pf` 是 `[BH,BT]` |
| `BLOCK_T` 16 → 32 | 45 056(**更大**) | 20.7 ms | 反而更差 |
| 合并 8 个累加器(前面测过) | **40 960(没变)** | 23.6 ms | **无效** |
| 绕开 `tl.dot` 用广播(前面测过) | 20 480 | 163.7 μs vs 61.6 | **不可行** — 慢 2.6× 且 fp16 溢出 |

`num_stages` 这一项值得单独说:它**早就被扫过时间**(21.19/21.17/21.22 ms,无差别),
我据此判断"stages 旋钮无效",但**从来没测过它对 smem 的影响**。现在测了:
1/2/3/4 全部是 40 960 B。所以 smem 不是流水线深度决定的,是**同时存活的 dot 操作数
个数**决定的 —— 而这个数被 8 个未展开的 QK dot + 8 个 PV dot 写死了。

**这指向唯一还没试的结构性手段:把 8 个 QK dot 放进一个真正的循环(不是展开),
让 Triton 复用同一块操作数暂存区。** 注意这与"合并累加器"不同 —— 那次只合并了
累加器,dot 本身仍是 8 个展开的,所以 smem 没动。这次要动的是 dot 的生成方式。

**风险提示:** 试过一次之后我对"看起来能省 smem"这件事保持怀疑。上面的数据说明
Triton 在 sm_75 上没有为这种 kernel 做过 smem 复用优化,即使写成循环也可能仍然
展开(编译期 `tl.static_range` 必然展开;改成 `range` 则 NCHUNK 变成运行时值,
而 dot 的形状必须是编译期常量)。**所以这条路能否成功,取决于 Triton 是否支持
"形状编译期已知但循环运行时执行"** —— 这是可以在 10 分钟内验证的,不用写完整 kernel。

## 第四个杠杆也测完了:smem 在 Triton 里降不下来(2026-09-29)

把 8 个 QK + 1 rope + 8 个 PV dot 从 `tl.static_range`(编译期展开)改成 `range`
(运行时循环,形状仍是编译期常量),在完整的 17-dot 结构上对比:

| 变体 | shared | FMA 指令数 | μs | 输出一致 |
|---|---|---|---|---|
| 全部展开(现状) | 4096 | 2304 | 57.0 | — |
| QK/PV chunk 用 `range` | **5120(更大)** | **512** | 55.9 | **逐位相同** |

Triton 确实把 `range` 变成了真循环(FMA 指令数降到 1/4.5,省了指令展开),
**但 smem 反而涨了 1024 B,时间基本没变(2%)。**

这说明 smem 的分配不是"同时存活的 dot 操作数"决定的,而是 **Triton 为
`tl.dot` 在无 `mma.sync` 路径上固定分配的下划线缓冲**,与展开方式、循环方式、
stages、累加器数量都无关。

## 四个杠杆的最终结论

| 手段 | smem 变化 | 判定 |
|---|---|---|
| 合并 8 个累加器 | 0 | 无效 |
| 绕开 `tl.dot`(广播乘法) | −50% 但慢 2.6× | 不可行 |
| `num_stages` 1–4 | 0 | 无效 |
| `BLOCK_T` 16→8 | 编译失败 | 不可行 |
| `BLOCK_T` 16→32 | +4096 | 变差 |
| QK/PV 改 `range` 循环 | +1024 | 无效 |

**六次尝试,全部没能降低这个 40 KB。** 每次都是先有假设、再被数据否掉。
我现在对"Triton 内部能降低这个 kernel 的 smem"这个命题**没有信心了**,
继续试第 7 个组合的期望收益很低。

## 对 2400 tok/s 目标的诚实评估

目标本身是合理的(需要 attention 143 → 70 ms,即 2x),但**我目前没有找到能达成它的手段**。

已经确定的物理约束(都有测量):
- attention 的 98% 是 FFMA 算术,gather 只占 2%
- 一个 block 的 dots 是算术地板的 14.4 倍,因为每 SM 只有 4 个 warp
- 要提 occupancy 必须降 smem,而 6 次尝试都没降下来
- 即使 attention 做到算术地板,端到端上限也只有 3097 tok/s

**剩下的唯一有机制依据的方向是绕开 Triton,写 CUDA/PTX kernel**,自己控制
smem 分配和 warp 调度。这不是"调参"或"改结构",是写一个 CUDA kernel——
工作量通常以周计,而且我无法保证 sm_75 上手写 FFMA 循环能超过 Triton 生成的版本
(Triton 至少把指令调度做对了:`range` 变体省了 4.5 倍 FMA 指令却没变快,
说明瓶颈在 smem 往返而不在指令数)。

**我的建议:不要在这一点上继续投入我一个人能推进的范围。** 可选:

1. 把这份分析完整保留(已经写在这里),作为将来做 CUDA 重写时的起点和避坑清单
2. 接受当前 1429 tok/s,把 opt4 的 +7% 固化成果交付(已完成)
3. 如果一定要追 2400,需要的是一个熟悉 sm_75 PTX 的人,或者直接上 CUDA C++

---

# 负载均衡:第一个不需要改 kernel 的正收益 — 2026-09-29

## 数据

sglang 有现成的开关 `SGLANG_PP_LAYER_PARTITION`(`get_pp_indices` 直接按它切层,
不均衡时剩余的层给靠后的 stage),所以这不需要改任何代码。

同池(210000)交替两轮,**唯一变量是层切分**,每轮重启并把环境变量从运行进程读回来核对:

| prompt | 10,11,11,11(现) | 11,11,11,10 | 比值 |
|---|---|---|---|
| ~11 800 | 1370.4 | **1473.0** | 1.075x |
| ~19 100 | 1418.2 | **1545.5** | **1.090x** |

轮间离散 A 0.07% / B 0.34%,差异远超噪声。**+8.2%**。

## 为什么有效

实测每 stage 每 chunk(同一 trace、同一口径):

| stage | 层数 | 每 chunk | 每层 |
|---|---|---|---|
| PP0 | 10 | 215.2 ms | 21.5 ms |
| PP1 | 11 | 254.8 ms | 23.2 ms |
| PP2 | 11 | 247.4 ms | 22.5 ms |
| **PP3** | 11 | **274.6 ms** | **25.0 ms** |

PP3 是瓶颈,而它每层比 PP0 贵 16%。43 层不能被 4 整除,现切分 `10,11,11,11`
让最贵的 stage 背了 11 层。`11,11,11,10` 把最轻的 stage 放到 PP3,
瓶颈从 274.6 ms 降到约 250 ms。

## 但这个配置有一个已知风险,不能直接上线

`11,11,11,10` 下 **PP0 背 11 层,decode graph 捕获后只剩 0.42 GB**
(现切分 PP0 背 10 层,余 0.74 GB;同样 11 层的 PP1/PP2 有 1.40/1.34 GB)。
差值的来源是 **PP0 额外持有 `embed_tokens`(129280x4096 fp16 = 1010 MiB)**,
PP3 额外持有 `lm_head`(同样 1010 MiB)。

实测后果:在 pool=270000 下跑 200 题 GSM8K,PP0 在 `ncclAllGather` 处
`unhandled cuda error` 崩掉,服务全挂。

**我一度以为缩 KV 池(pool=210000)能解决,这是错的** —— 缩池对 PP0 的 avail mem
几乎无影响,因为 PP0 的瓶颈是它多背的 embed_tokens + 11 层权重,不是 KV 池。
210k 那次能跑完 200 题(94.0%、0 invalid)只是运气好,余量问题依然存在。

**要安全落地 11,11,11,10,必须先解决 PP0 的 1010 MiB embed_tokens 占用。**
可行方向:把 embedding 单独放一个 stage(PP4 但只有一个 KV 对?)、或用
`--enable-torch-compile` 之类无关;更实际的是接受 `10,11,11,11` 并从别处找收益。

## 这次的价值

1. **+8.2% 已经确认存在,且可复现**,是迄今为止除了 MoE 修复之外最大的单项收益
2. 它**没有碰任何 kernel**,说明之前的注意力全放在 attention 上是个盲点
3. 它暴露了一个之前没意识到的约束:**PP0 背 embed_tokens 让她比其它 stage 贵 1 GiB**,
   这限制了层切分的自由度

## 与 chunk 的关系(同一个教训)

同一次实验里也测了加大 chunk:chunk=1024(pool 230k)**能跑**,
但 tok/s 与 512 相同甚至略低(1419 / 1410 / 1411)。
我之前推理"每 chunk 90 ms 气泡,chunk 加倍能摊薄省 12%"是**错的** ——
那 90 ms 是 pipeline skew,正比于每 chunk 的计算量,不是固定开销。
流水线已经 95% 满载,加大 chunk 只会让气泡等比例变大。

---

# 11,11,11,10 为什么不能上线:PP0 的显存是硬约束 — 2026-09-29

## 需要的余量

实测每层权重(safetensors metadata 精确统计,不用估算):

```
43 层共 146 768 MiB  ->  每层 3413 MiB
TP2 分摊后每卡每层   =  1706 MiB
embed_tokens = 1010 MiB   lm_head = 1010 MiB   mtp = 10 360 MiB
```

| stage | 现切分 | 权重/卡 | 改后 | 权重/卡 |
|---|---|---|---|---|
| PP0 | 10 层 + embed | 17 570 MiB | **11 层** + embed | **19 276 MiB** |
| PP3 | 11 层 + mtp + head | 48 913 MiB | **10 层** + mtp + head | 45 500 MiB |

**PP0 加一层要多占 1706 MiB/卡**,而它改后权重已占卡的 85.6%。
实测 decode graph 捕获后余量:0.74 GB(10 层)→ **0.42 GB**(11 层),
200 题 GSM8K 在 `ncclAllGather` 处 `unhandled cuda error` 崩溃。

## 为什么缩 KV 池救不了

我先怀疑是 KV 池太大。实测否掉了:把 pool 从 270000 降到 210000,
PP0 的 avail mem **仍是 0.42 GB**,一点没改善。

原因是 sglang 的 `mem_fraction_static` 会把权重增加的部分从 KV 池里扣掉,
KV 池已经缩到底(pool=210000 时它没法再缩,因为要保证 decode batch 有位置),
于是余量就实打实地少了。

## 为什么降 mem-fraction 也救不了 —— 硬下限

这是决定性的一条。`--mem-fraction-static 0.94` 直接被 sglang 拒绝:

```
ValueError: Loaded weights leave no GPU memory for the KV cache under
--mem-fraction-static=0.94. Raise --mem-fraction-static above 0.955
(minimum viable = 1 - available/pre = 0.9548).
```

**可用区间只有 0.9548 – 0.97**,即最多能多要
`0.0152 x 22528 = 342 MiB/卡`,而缺口是 **1706 MiB**。差 5 倍。

## 结论

`11,11,11,10` 在这台机器上**无法安全启用**。43 层分给 4 个 stage 时,
唯一能让 PP3(瓶颈)少背一层的分配就是让 PP0 多背一层,而 PP0 已经贴着显存墙。

**除非能减掉 PP0 的 embed_tokens 占用(1010 MiB),或者把 embedding 移出 PP0 ——
两者都需要改 sglang 的模型装载逻辑,不是配置能解决的。**

这条 +8.2% 的收益因此**记为"已确认但当前不可用"**,写在这里是为了将来
(如果 embedding 能跨 stage 放置或做 tied 复用)可以直接捡回来。

## 顺带纠正的一处

我一度以为"pool=210k 时 200 题跑完(94.0%、0 invalid)说明余量问题解决了"。
**那是错的** —— 那次跑的是 10 层切分,不是 11 层切分。11 层切分下的 PP0
无论 pool 多少都是 0.42 GB。同一份日志里 0.68 GB 那一档来自 10 层的 PP0。

---

# MoE 与 gemm 的调查 — 三个都是死路(2026-09-29)

上一节说"MoE 62 ms + gemm 35 ms = 46% 从没碰过"。查完了,结论是**这三个方向
都没有可用的空间**,原因各不相同,而且都有实测支撑。

口径:PP3 三个 chunk 全部,union 去重叠,共 871 ms 计算。

| 家族 | ms | 占比 | 判定 |
|---|---|---|---|
| attn | 440.2 | 50.5% | Triton smem 锁死 occupancy(前六次已否) |
| **moe** | 206.6 | 23.7% | **带宽 58%,内存受限** |
| **gemm** | 120.7 | 13.9% | **与 cuBLAS 持平,无空间** |
| elem | 81.2 | 9.3% | 最多省 2-3% |

## moe:w4a16_v3_kernel 是内存受限,不是实现问题

`w4a16_v3_kernel` 20 次 60.45 ms(单 chunk 口径)。它已经是**手写 PTX
`mma.sync.aligned.m16n8k8`**(不是 FFMA),所以它在用 tensor core。

双重屋顶(每卡 128 个 expert,MXFP4 = 0.5 B/param):

```
每 expert 权重 8 MiB(w13 6 MiB + w2 2 MiB)
每次调用 128 个 expert 全读 = 1024 MiB
带宽屋顶 = 1024 MiB / 616 GB/s = 1.74 ms   实测 3.02 ms  -> 58% 带宽
算力屋顶 = 2.9 GFLOP / 59.2 TFLOP/s = 0.05 ms            -> 算力不是限制
```

**arithmetic intensity 只有 0.05 FLOP/byte,而硬件 ridge point 是 96。**
差 2000 倍 —— 这是彻底内存受限。

根因:512 token × topk 6 = 3072 行,铺满 128 个 expert = **每个 expert 只有 24 行**,
却要把 8 MiB 权重整个读进来。**读权重的时间正比于激活的 expert 数,与 token 数无关。**

要提速只能减少权重读取量,而 prefill 就是 512 token,128 个 expert 必然全被激活。
唯一理论解是权重驻留(1024 MiB),但卡上余量只有 0.42–0.74 GB,放不下。

## gemm:sglang 和 cuBLAS 一样快

最大的 gemm 是 `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn`,231 次 53.57 ms,
单次 0.232 ms。我用 cuBLAS 复现候选形状:

| 形状 | cuBLAS | trace 实测 |
|---|---|---|
| `[512,2048]×[2048,2048]` | **0.230 ms** | **0.232 ms** |

**完全吻合。** 也就是说 sglang 在这个形状上没有任何额外开销,已经是 cuBLAS 水平
(18.5 TFLOP/s = fp16 峰值的 31%)。

31% 看起来低,但**cuBLAS 自己也只有 18.7 TFLOP/s**。这是**形状本身太小**
(M=512,tile 数有限)导致的,不是实现问题。换成更大的形状,cuBLAS 能到 40 TFLOP/s
(`[512,4096]×[4096,4096]` 0.429 ms),但模型结构决定了 prefill 就是这个形状。

## elem:唯一有点空间的,但上限 2-3%

81.2 ms / 2337 次调用。发现一个真实的低效:

```
grid=[1,1,1] 的 kernel: 501 次,单 block
  -> 68 个 SM 里只用了 1 个
  -> 但合计只有 1.76 ms,占 1.2%
```

即使全部优化到零也只省 1.2%。最大的两个 elem(`grid=[253]` 273 次 21.28 ms、
`grid=[48]` 252 次 17.52 ms)已经是 3.7 blocks/SM,不算太糟。

乐观估计 elem 能省 20-30%,即 16-24 ms,**端到端 2-3%**。

## 总结:注意力不该放在这上面

三个方向都查完了,**没有 5% 以上的空间**:

- MoE:内存受限,读权重是硬成本
- gemm:已达 cuBLAS 水平
- elem:零散小 kernel,总量小

**加上之前排除的**(attention smem 六次、chunk 加大、拓扑/NCCL、mem-fraction、
PP 层重排的显存墙),这台机器上我已经**没有找到任何一个能带来 5% 以上收益、
且风险可接受的方向**。

唯一还有较大理论空间的是 attention 的 440 ms(50.5%),但它需要绕开 Triton 写
CUDA/PTX kernel —— 我在前面已经说明这不是调参能解决的。

## 一次方法论上的教训

这一轮我三次差点得出错误结论,都是因为**用估算代替实测**:
1. 猜 gemm 形状算出 113 TFLOP/s(超过硬件峰值 59.2)—— 形状猜错了;
   用 cuBLAS 复现才对上
2. 说"gemm 是启动开销主导"—— 实际中位数 0.185 ms,不是小 kernel
3. 说 `grid=[1,1,1]` 的 651 次占 10.2 ms —— 实际是 501 次 1.76 ms,
   另 150 次属于别的 grid

**每次都是先算一个数,发现不合理才去测。** 应该反过来:先测,再解释。
