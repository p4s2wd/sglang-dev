# PROGRESS — DeepSeek-V4-Flash on 8× RTX 2080 Ti (SM75)

本文件记录本机(8× RTX 2080 Ti,每卡 22 GiB,TP2×PP4)的**已发布里程碑**。
每个里程碑对应一个 GitHub release,内容包括根因、验证数据和踩过的坑。

- 仓库:`https://github.com/p4s2wd/sglang-sm75`(sglang 的 fork,`main` 保持上游状态,内容全在 release 附件里)
- 本地 release 目录:`/data/nvme/sglang/release-sm75/`
- 服务:`/data/nvme/sglang/deepseek-v4-flash.sh`,端口 8200,日志 `logs/serve-prod.log`
- 早期设计文档:`sglang-sm75-plan.md` / `sglang-sm75-progress.md` / `sglang-sm75-runbook.md`

---

## 发布索引

| 版本 | Tag | Release | 一句话 |
|---|---|---|---|
| opt1 | `v0.1.0-sm75opt1` | *已从 GitHub 下架(见文末「GitHub release 收敛」)* | sub-90 性能优化:indexer 快速路径、fp16 MQA logits cuBLAS 改写、decode seq-len 分桶、W4A16/W8A16 内核 |
| opt2 | `v0.1.0-sm75opt2` | *已从 GitHub 下架(见文末「GitHub release 收敛」)* | 修 opt1 的两个正确性 bug(indexer logits 溢出成 `inf`、Vision-Exp 加载 `KeyError`)+ 修好补丁文件 |
| opt3 | `v0.1.0-sm75opt3` | *已从 GitHub 下架,被 opt4 取代(2026-09-29)* | 修真正让 pi 报错的那件事:默认采样 temperature=1.0 往长 tool call 里塞野 token |
| **opt4** | `v0.1.0-sm75opt4` | **[链接](https://github.com/p4s2wd/sglang-sm75/releases/tag/v0.1.0-sm75opt4)** | prefill 吞吐 **+6.4%/+7.7%**(交错 A/B):MoE combine 从扫全部 padded slot 改为 per-token slot 表 |

> **只有 opt4 在 GitHub 上发布。** opt1 / opt2 / opt3 的 release 与 assets 已删除,四个 git tag
> 都保留;本地归档 `release-sm75/superseded/`(opt1/opt2)与 `release-sm75/`(opt3)未动,重建链
> 仍可用。**opt4 是累积的**:opt1 的性能改动 + opt2 的两个正确性修复 + opt3 的采样旋钮 +
> opt4 的 prefill 修复,单装它即可。详见文末。

---

## opt4 — 2026-09-28(已发布 2026-09-29)

**Release:** https://github.com/p4s2wd/sglang-sm75/releases/tag/v0.1.0-sm75opt4
**本地产物:** `/data/nvme/sglang/release-sm75-opt4/`
- `sglang-0.5.19.dev332+g60b7c400e7.sm75opt4-py3-none-any.whl` (`4c600bf3…`)
- `sglang-sm75opt4-release.tar.gz` (`e34b7681…`)
- 改动文件:`kernels/ops/elementwise/elementwise.py`(+102 / −17),**在 opt3 之上累积**

**根因:** `_moe_combine_kernel` 找 token 拥有的 expert 行时内层是
`for s in range(0, num_slots)`,每个 program 扫全部 padded slot,而它只拥有 `topk`(=6)行。
prefill 形状(4096 programs × 3072 次)下是 1250 万次标量 load,只为了取 3072 行;
该 kernel 占 PP3 device 时间 9.4%。decode 下 `num_slots` 只有 ~8×topk,所以缺陷此前未暴露。

**修法:** 先把 slot→owner scatter 成 `[num_tokens*topk]` 表(缺失填 −1),combine 再按
`topk` 次迭代取行。两条路径按形状自动切换,门槛 `num_tokens >= 256 and num_slots >= 28*topk`
来自实测交叉点(B=128 是 0.48x,B=512 是 2.85–3.84x)。全程 graph-safe(不用 `nonzero()`)。

**效果(交错 A/B,两轮,每轮之间重启服务):**

| prompt tokens | opt3 | opt4 | 提升 |
|---|---|---|---|
| ~11 800 | 1272.8 | 1353.9 | **+6.4%** |
| ~19 100 | 1312.7 | 1413.9 | **+7.7%** |

同一 arm 轮间离散度 0.8–1.2%。单次前后对比在这台机器上不可信——同一构建两次测量给出
1429 与 1366 tok/s(GPU 已连续负载数小时),所以必须交错测量。

**只拿到 7% 而不是 30% 的原因:** 瓶颈**转移**了。PP3 每 chunk 计算量没下降
(280.7 → 282.3 ms),MoE 占比 30.4% → 22.6%,而 attention 占比升到 52.2%。

**验证:** `moe_combine` 26 用例全过(B=1/2/8/64/256/512,topk=1/6/8,
hidden=512/576/1024/2048/4096,含 `num_valid` 只覆盖一半的情形),并对 **wheel 内的副本**
重跑同样 26 个用例;`RECORD` 4457 个文件全量校验;patch `patch -p1` 正反向均可应用且
打完与 wheel 逐字节一致;GSM8K 200 题 5-shot 贪心 93.5%、0 invalid(opt3 全量基线 94.2%)。

**过程中被数据否掉的版本:** 第一版 slot 表用「每个 (token,k) 各扫一遍 num_slots」构建,
结果**慢 2 倍**。拆开计时才看清 combine 本体 0.649→0.084 ms(7.7×),是构建器要 1.29 ms;
改成 scatter 后构建器降到 0.09 ms。只看端到端数字会误判为「负收益」。

**同时挖出两个 prefill profile 结论**(详见 `PREFILL_PROFILE.md`):

1. **流水线已经 ~90% 满载**,不是现有笔记里说的 29–30%。调度最多再买 1.34×,到不了
   3500 tok/s。现有那些数字来自 9-20 对 dev 服务器(:30000,已死)的 capture,在 opt1 内核
   之前,已不成立。
2. **`SGLANG_SM75_HS_NCOL=32` 是静默错误的**(看起来 1.725×,实际 256/256 token 全错)。
   `NCHUNK = 512//NCOL = 16` 超过 kernel 展开的 8 个 chunk,448 维 nope 的上半部分从
   QK 和 PV 里同时消失。生产不设这个变量(默认 64)所以没出事;`ab_prefill.sh` /
   `ab_prefill2.sh` 从未跑完(无 `logs/ab-*` 输出)。**任何后续调这个 kernel 的人必须先
   确认 `512 // NCOL <= 8`。**

---

## opt3 — 2026-09-28

**Release:** https://github.com/p4s2wd/sglang-sm75/releases/tag/v0.1.0-sm75opt3
**产物:** `sglang-0.5.19.dev332+g60b7c400e7.sm75opt3-py3-none-any.whl` (`f54df558…`),
`sglang-sm75opt3-release.tar.gz` (`ede55b98…`)

### 根因

pi 这类客户端不发 sampling params,服务端回落到 checkpoint 的 `generation_config.json`,
那里是 DeepSeek 的通用占位值 `do_sample: true, temperature: 1.0, top_p: 1.0`。**完全不截断的
全熵采样**,对几千 token 的 tool-call 参数是灾难:sglang 的 `--sampling-defaults` 两个取值
(`model` / `openai`)都给出 1.0,**服务端没有任何旋钮能改**——这才是缺的那块。

复现并量化(9138 token prompt + 4000–6300 token 生成,`/generate` 原始 DSML,统计剔除注释和
字符串后"裸代码"里出现的 CJK / 带音标拉丁字符):

| 采样 | 长代码生成 | 结果 |
|---|---|---|
| `temperature=0.0` | 3 次 | 全干净,**逐字节相同** |
| `temperature=0.6` | 3 次 | 全干净 |
| **默认 1.0 / 1.0** | 9 次 | **2 次被插野 token** |

被插进去的形态,与用户会话里模型自己 grep 的字符串同型:

```js
const H = canvas.height[廳];        // 野 token 楔进下标 → JS 语法错误
if (dx > 40) dir += 0.5[笠];        // 野 token 贴在数字字面量后
if (dx < -40) dir -= 0.5笠;
```

一次野 token 同时造成三件事:文件语法错误、tool call 参数无法解析、模型在下一轮看到自己上一轮
"写坏了"的历史而反复重写(原话 "I keep producing garbage — stray words inserted like
'unexplained', 'ferential', '560', '679'")。

### 修法

新增 `SGLANG_DEFAULT_SAMPLING_PARAMS`(JSON,覆盖 `temperature / top_p / top_k / min_p /
repetition_penalty`),落进与 `model_generation_config` 同一个 dict,优先级为
**请求里的值 > 环境变量 > generation_config.json > sglang 默认**——客户端自己指定 temperature
的永不受影响。**默认不设,行为与 opt2 完全一致。**

> 参考机的实际处置:把 checkpoint 的 `generation_config.json` 改成 `temperature: 0`(2026-09-28
> 15:14 由用户完成),环境变量是给不想动共享 checkpoint 的人准备的备用旋钮。

改动文件:`srt/entrypoints/openai/serving_chat.py`(+77 行)、`launch-dsv4-sm75.sh`(仅注释)。

### 验证

pi 直连 8200(不经 newapi)跑 3 个完整任务,零野 token,`node --check` 全过:

| 产物 | 大小 | 扫描 | JS 语法 |
|---|---|---|---|
| `space-shooter.html` | 16041 ch | CLEAN | OK |
| `tetris.html` | 14159 ch | CLEAN | OK |
| `accounting.html` | 19693 ch | CLEAN | OK |

### 排除了什么(避免重复踩)

- **chat template 没坏**——渲染出的 prompt 结构正确、4 个 tool schema 齐全、
  `reasoning_content` 正确回到 `<think>`、DSML 往返一致;唯一差异是 pydantic
  `model_dump()` 给每个工具多加了 `"strict": false`,无害。
- **tool-call 解析器忠实**——模型原始 DSML 13831 → 解析后 13830 字符,只少一个结尾换行。
- **长 decode 是确定性的**——同 prompt + temperature 0 三次生成 md5 相同。

**坑在于探针选错**:用"逐字复制给定文本"当指标,分布太尖,温度 1.0 也能过;原创代码熵高得多,
必须用长代码生成来测。GSM8K / NIAH 同理——都是短输出,照不到这个 bug。opt2 的验收指标里
缺的正是这一项。

---

## opt2 — 2026-09-28

**Release:** *已下架*(2026-09-28 删除;本地归档 `release-sm75/superseded/`)
**产物:** `sglang-0.5.19.dev332+g60b7c400e7.sm75opt2-py3-none-any.whl` (`e79b63e0…`),
`sglang-sm75opt2-release.tar.gz` (`58b16688…`)

### 1. indexer logits 溢出成 `inf` → 长上下文输出退化

`kernels/ops/attention/dsa/triton_mqa_logits.py` 用 cuBLAS fp16 tensor-core GEMM 算 MQA
logits(sub-80 无 DeepGEMM)。indexer 的 q/k 是 **absmax 缩放到 448** 量化
(`main_norm_rope.cuh`:`scale = absmax / FP8_E4M3_MAX`),D=128 的点积最大
`128 × 448 × 448 = 2.6e7`,而 fp16 上限 65504,`torch.mm` 返回 fp16 → **整块 logits 饱和成
`inf`**,top-k 退化成随便取一段前缀,稀疏注意力选错 key。

低于约 2K 上下文看不出来(c4 长度还没超过 `index_topk=512`,top-k 无论如何全选),而 coding
agent 的 system prompt + 代码很容易超过 4K —— 所以线上一直在报错。

修法:两个 GEMM 前各除掉一个 2 的幂(fp16 里精确,relu 与正缩放可交换),逆缩放在 fp32 还原;
q 的上界只用 fp8 动态范围(编译期常量,比数据相关上界更稳也更省),w 的上界一次 reduction;
顺带把 logits 尾部 page padding 从 `new_empty` 残留改成显式清零。相对误差 ~5e-4,比上游同硬件
的 bf16 torch fallback(~4e-3)还好,速度比原 Triton 内核快 ~20x。

### 2. Vision-Exp 权重加载 `KeyError`

`remap_weight_name_to_dpsk_hf_format` 无条件把 `.w1./.w2./.w3.` 改写成 MoE 投影名。Vision-Exp
带了一个 vision aligner(同样是 w1/w2 的 MLP),被改写成 `aligner.gate_proj.weight` 后命中
stacked `gate_up_proj` mapping,`params_dict[...]` 抛 `KeyError`,整个 load 挂在第 0/48 个 shard。

修法:把 w1/w2/w3 改写限定在 MoE 路径(`experts`);stacked mapping 查不到参数时 warn+skip 而不是
KeyError。Vision-Exp 现在能起来(dtype 自动 bf16→fp16,权重占用与 0731 相同,145.3 GiB),但以
`has_image_understanding: false` 提供**纯文本**——sglang 没有 DeepSeek-V4 的 vision 实现。

### 3. `sm75-optimizations.patch` 修好了

opt1 用 `diff -ruN` 生成,表头是 `--- x/sglang/...`,文档里写的 `patch -p1` **实际一个 hunk 都
打不上**。opt2 换成 `a/python/sglang/...` / `b/python/sglang/...`,`patch -p1` 与 `patch -R -p1`
均验证通过,且打完的结果与 wheel 内文件逐字节一致。

### 验证

| | opt1 | opt2 |
|---|---|---|
| NIAH 6K / 12K / 30K 首 token 吐 EOS | 7/8, 2/12, 2/12 | **0/8, 0/12, 0/12** |
| GSM8K(0731,100 题 5-shot 贪心) | — | **96%** |
| GSM8K(Vision-Exp,100 题) | 加载失败 | **96%** |
| 0731 加载告警 | 0 | 0 |

---

## opt1 — 2026-09-25

**Release:** *已下架*(2026-09-28 删除;本地归档 `release-sm75/superseded/`)

sub-SM80 性能优化基线:prefill indexer 快速路径真正生效、index logits 改 cuBLAS fp16
tensor-core GEMM(19 ms → 1.5 ms/层)、decode seq-len 分桶、Triton top-k transform、
W8A16 wide-GEMV 开关、可选 PP 提前发 proxy、chat 模板加 `add_default_bos_token=False`。
详见 release notes。

> 注:opt1 引入的 `add_default_bos_token=False` 是**必需**的,不是多余的。带 BOS 时模型会一直复读
> `<｜begin▁of▁sentence｜>` 直到耗尽 token budget(opt3 下仍然如此),且 tokenizer 自身
> `add_bos_token: false`,所以别处也不会补一个。

---

## 遗留 / 待办

- **attention 是下一个 prefill 瓶颈(占 PP3 device 时间 52.2%)。** 要到 3500–5000 tok/s,
  需要 PP3 每 chunk 计算量从 ~282 ms 降到 100–145 ms,即真实计算量砍 2–2.8 倍。
  已排除的手段:launch 旋钮(`SGLANG_SM75_HS_NCOL/WARPS/STAGES`)在 prefill 形状 B=512 上
  全扫过,现值已最优,`stages` 交错 5 轮实测无差别;occupancy 卡在共享内存
  (40 KB/block 对 64 KB 上限只容 1 个 block,降寄存器无效),而唯一能降低累加器占用的
  方向(更小 NCOL)正好是上面那个正确性陷阱。剩下三种结构性改法:合并为单个
  `[BLOCK_H, 512]` 累加器、绕开 shared memory staging、或改写成显式 FFMA 不用 `tl.dot`
  (本 Triton 构建在 sm_75 上把 `tl.dot` 全降级为标量 FFMA,PTX 验证 0 条 `mma.sync`、
  2048 条 `fma.rn`)。都是真正的 kernel 重写,不是调参。
- **opt4 已发布,opt3 已被替换。** GitHub 上只剩 opt4 一个 release(14 assets)。四个 tag
  (`opt1`/`opt2`/`opt3`/`opt4`)全部保留,`main` 分支未推任何代码。
- **51 个本地 sm75 提交仍未推上任何远端。** 它们是 opt1–opt4 所有改动的真实来源,但目前
  只存在于本地 checkout 和 `backup/*.bundle`(共 ~150 KB)。要重建任何一个版本,只能用
  release 里的 wheel 或 patch,不能用 git。要不要推分支到 fork 还没定。
- **`num_nextn_predict_layers: 3`(Vision-Exp)与 nextn loader 的 `assert == 1` 冲突**,
  开投机采样前要先处理。
- **vision 输入未实现**:Vision-Exp 的 vision 塔与 aligner(0.87 GiB)被跳过,图片输入会被拒。
  要支持需要给 sglang 写 DeepSeek-V4 的 vision 模块。
- **fp16 而非 bf16** 是 SM75 的硬限制(无原生 bf16);runbook 记录为已知取舍。
- **重新构建链现在是 base → opt1 → opt2 → opt3 → opt4**。原始 base wheel 不在本机,
  所以 opt4 是在 opt3 wheel 上重建的(opt3 是累积的,内容等价)。
- 基座 wheel 的 sha256(`8f0a24c4…`,文档里引用)无法在本机核对——原始 base wheel 不在这里。
- **`/start_profile` 会打死服务端**(上游 bug):`profile_by_stage=True` 配
  `num_steps=0` 时 `profiler_prefill_ct` 保持 `None`,`profiler_manager.py:417` 做
  `None += 1` 抛 TypeError,8 个 rank 全挂。`prof_prefill_prod.py` 已在前置检查里拒绝该组合。

---

## GitHub release 收敛 — 2026-09-28 / 09-29

**当前状态(2026-09-29):只剩 opt4 一个 release。** 历史上分两次收敛:

- 2026-09-28:只保留 opt3,删除 opt1 / opt2(见下)
- 2026-09-29:opt4 取代 opt3,删除 opt3 的 release 与 assets

两次都遵循同一条纪律:**先建好新的、验证它能独立成立,再删旧的**,避免出现
"删完一个都不剩"的中间态。

### opt4 取代 opt3 的前置校验(2026-09-29)

| 检查 | 结果 |
|---|---|
| `gh release view v0.1.0-sm75opt4` | draft=false, prerelease=false, **14 个 assets** |
| 重新下载 assets 后 `sha256sum -c SHA256SUMS` | **11/11 OK** |
| tarball sha256 vs `MANIFEST.sha256` | `e34b7681…` == `e34b7681…` |
| 装进干净 prefix 后的行为 | version `…sm75opt4`;opt4 slot 表 / opt3 采样旋钮 / opt2 indexer 与权重改名修复全部在位;`sm75_extras/launch-dsv4-sm75.sh` 存在 |
| `sm75-optimizations.patch` 对基座源码树 | `patch -p1` 正向可应用,打完与 wheel 内文件**逐字节一致**(0 处不符) |
| 删除 opt3 后复查 opt4 | assets 仍 14 个,`sha256sum -c` 仍全部 OK |

**发布前还修掉了一个文档缺陷:** opt4 的 `RELEASE_NOTES.md` 最初是重写而非增补,导致
README 里指向 opt2/opt3 章节的锚点(`#what-opt2-fixes`、`#what-opt3-fixes`、中文版两个)
全部失效。改成在 opt3 的双语 notes 上**前置 opt4 章节**(中英各一节),并修正
assets 表里遗留的 `sm75opt3` 缩写,现在 6 个锚点全部可解析。

### 第一次收敛(opt1/opt2 → opt3,2026-09-28)

只保留 opt3 一个 release,opt1 / opt2 连同 assets 一起删除。执行前先确认 opt3 已经发布且完整,
避免出现"删完一个都不剩"的中间态。

### 删除前的前置校验(opt3 自身可独立成立)

| 检查 | 结果 |
|---|---|
| `gh release view v0.1.0-sm75opt3` | draft=false, prerelease=false, **14 个 assets** |
| 重新下载 14 个 assets 后 `sha256sum -c SHA256SUMS` | **11/11 OK**(SHA256SUMS 不含自身) |
| tarball sha256 vs `MANIFEST.sha256` | `ede55b98…` == `ede55b98…` |
| 装进干净 prefix 后的行为 | version `0.5.19.dev332+g60b7c400e7.sm75opt3`;`SGLANG_DEFAULT_SAMPLING_PARAMS` 覆盖生效(opt3);`_plan_scales` 在位(opt2);`experts` 限定改写生效且 `aligner.w1` 不再被改写(opt2) |

opt3 是累积的:opt1 的性能改动 + opt2 的两个正确性修复 + opt3 的采样旋钮,单装它即可。

### 执行

```
gh release delete v0.1.0-sm75opt1 --repo p4s2wd/sglang-sm75 --yes
gh release delete v0.1.0-sm75opt2 --repo p4s2wd/sglang-sm75 --yes
```

### 验证 —— `gh release list`

```
$ gh release list --repo p4s2wd/sglang-sm75 --limit 10
sglang SM75 optimizations for DeepSeek-V4-Flash (RTX 2080 Ti)  Latest  v0.1.0-sm75opt3  2026-09-28T11:12:13Z
```

只剩 opt3 一条,且它是 `Latest`。逐个复查:

```
v0.1.0-sm75opt1   (no release)
v0.1.0-sm75opt2   (no release)
v0.1.0-sm75opt3   v0.1.0-sm75opt3
```

删除后再次确认 opt3 未受影响:assets 仍为 **14** 个,本地 `sha256sum -c SHA256SUMS` 仍 **11/11 OK**。

### 有意保留 / 未动的部分

- **git tag 保留**:`v0.1.0-sm75opt1` / `v0.1.0-sm75opt2` / `v0.1.0-sm75opt3` 三个 tag 都还在远端。
  任务只要求删 release(含 assets),tag 与 release 是两回事,留着 tag 便于日后按同一 tag 重新切
  release。需要一并清掉的话:`git push origin --delete v0.1.0-sm75opt1 v0.1.0-sm75opt2`。
- **本地 `release-sm75/superseded/` 一字未动**:opt1 / opt2 的 wheel 与 tarball(mtime 仍是
  2026-09-25 与 2026-09-28 14:30 / 14:32),重建链 base → opt1 → opt2 → opt3 仍然走得通。
- 仓库 `main` 分支未推任何代码,内容依旧全在 release 附件里。

### 对使用者的影响

opt1 / opt2 / **opt3** 的 release 页面与 assets 链接现在都是 404。**唯一可下载的是 opt4,
它是累积的,单装即可。** 需要旧版本时:

- opt3:本地 `/data/nvme/sglang/release-sm75/`(wheel + tarball 齐全,校验和完整)
- opt1 / opt2:本地 `release-sm75/superseded/`

四个 git tag 都在远端,但它们指向的 commit 是**上游 sglang 的 commit,不含 sm75 代码**
(`opt1` → `2f5c9ac43` 是上游 Cosmos3 提交;`opt2`/`opt3`/`opt4` 同指 `cdbea5dcc`)。
所以 tag 无法用来重建 release——**只有 wheel 和 `sm75-optimizations.patch` 能重建**。
这一点与上一版文档的说法相反,已更正。

---

## 2026-09-30 | #4 收尾:headshared 重写否决 → 转向 allreduce 归因(哥定:方案 A)

### 结论速览

1. **headshared prefill kernel 没有重写空间**:旋钮已扫到最优(NCOL64/W4/S2),
   rowgather 变体(整行一次 gather)实测 **2.25x 慢**(48.4 vs 21.5 ms @ B=512/TOPK=512,
   输出还与 baseline 逐位相同),已从 `flash_mla_sm120_triton.py` 移除,baseline 恢复为与
   HEAD 逐位一致(rg3 harness:err 2.9e-04 vs fp32 ref,nan=0,decode 形状 0.625 ms 不变)。
2. "bench 20 ms vs 生产 2.8 ms"之谜 = **harness 合成数据 bug**:rope 字节位置用 page stride
   584 计算(token stride 实为 576),随机字节残留 → bf16 0x6464≈2^73 → `.to(f16)`= inf →
   `0×inf`=NaN。差点误判 baseline kernel 有 NaN bug。教训:**合成 cache 必须按
   [payload PAGE×576][scale PAGE×8] 页内布局逐 token 填,rope 用真实 bf16 值**(rg2/rg3.py 已修正)。
3. 生产 EXTEND trace(PP3):headshared 只占 device 时间 18.8%,**allreduce_1shot_push 占 44.6%**
   → #4 重定向到 allreduce。

### allreduce 归因(prof2b:profiles/20260930-153310-pf2b,probe 4.3k tok × 6 步)

**配对确认**:`nvidia-smi topo -m` 显示 NV2 链路正好是 (0,1)(2,3)(4,5)(6,7),rank 分配
(tp-first)使 TP2 组 = NVLink 对内 → **目标 allreduce 全部走 NVLink,不跨对;跨对只有 PP
激活 SendRecv(PCIe,5-13 ms/step,次要)**。

EXTEND burst 窗口内(剔除 profiler gap 后)各 rank:

| rank | burst | GPU busy | AR wait | wait/busy |
|---|---|---|---|---|
| TP0-PP0 | 147 ms | 86.8% | 73.1 ms | **57%** |
| TP1-PP0 | 146 ms | 33.4% | 0.5 ms | 1% |
| TP0-PP1 | 185 ms | 42.2% | 14.0 ms | 18% |
| TP1-PP1 | 182 ms | 49.4% | 26.0 ms | 29% |
| TP0-PP2 | 214 ms | 89.1% | 118.2 ms | **62%** |
| TP1-PP2 | 217 ms | 28.4% | 0.6 ms | 1% |
| TP0-PP3 | 205 ms | 70.8% | 69.6 ms | 48% |
| TP1-PP3 | 162 ms | 45.8% | 12.1 ms | 16% |

关键证据:

- **wait ≡ skew**:逐事件对齐两 rank 的 AR 启动时刻,偏移(+2.4 ~ +9.5 ms)≈ 领先侧的
  AR 自旋时长;落后侧 AR 真实耗时中位数 **25 µs**(peer 早已把数据推进本地 HBM)。
  慢侧 AR 忙时里 60% 是空转。
- 方向每对相反(PP1 是 TP1 早,其余 TP0 早)→ **不是静态的 rank 角色不对称,是无约束漂移**:
  EXTEND 前向是 eager(trace 里 graph id=0),CPU launch 速度决定 GPU 推进速度,每层段
  rendezvous 一次就暴露一次漂移。
- **wire 从来不是瓶颈**:≤16.8 MB payload 在 NV2(30 GB/s/dir)理论 ~0.6 ms;实测慢侧真实
  wire 工作 25 µs-0.9 ms(被领先侧的 push 覆盖)。
- 自旋 kernel 占满 **全部 68 个 SM ×1024 线程(100% occupancy)在 poll 本地 HBM** —— 
  150 W 功耗墙下,自旋还反过来拖慢领先侧自己的后续 kernel(反馈环,嫌疑已标记未定量)。
- kernel 机制(`custom_all_reduce.cuh:137`):Lamport one-shot push,每 rank 把数据 16B
   relaxed-store 到双方 slot(远端写恰好一次),然后本地 poll+reduce+清零。**已经是单向下发
  (单份拷贝)形态**。

### 三条路径的判定(基于上表证据)

| 路径 | 判定 | 理由 |
|---|---|---|
| 按 chunk 流水重叠 | ✗ 无效 | wait 是 rank 间 skew,不是带宽;wall time 由慢 rank 的 CPU 推进速度决定,重叠只是让领先侧空转变"有用功空转",不缩短 burst |
| 单向下发 | ✓ 已是现状 | Lamport push 每份数据只走一次 NVLink;再减只能减本地 slot 写,微优化 |
| 减少每层 AR 次数 | △ 仅省电 | 少 rendezvous=少暴露 skew 窗口,但 skew 会转移到残余 rendezvous;主要收益是自旋功耗(68×1024 全 SM 烧 150 W 墙)→ 建议把自旋改成小 grid+退避,顺手做 |

### 新的主要杠杆(取代"allreduce 带宽优化")

**EXTEND 是 CPU-launch-bound**:慢 rank burst 内 GPU busy 仅 28-49%,EXTEND 走 eager。
慢 rank 为什么更慢 / 两 rank 漂移的根因是下一个要挖的点(候选:eager python 逐层开销、
TP rank 间 CPU 差异、GC、PP 激活 PCIe 到达时刻的 convoy 级联)。
候选动作(未开始):prefill 侧 CUDA graph 分桶 / 降 launch 数 / 自旋小 grid 化 / 慢 rank CPU 采样。

### 复现与现场

- 分析脚本:`/tmp/opencode/pfshare2.py`(per-rank 统计)、`ar_burst.py`(burst 窗口 busy/wait)、
  内联 AR-start 对齐(`res[f"TP{t}-PP{p}-EXTEND"]` ts 差表)。
- 正确性 harness(合成 cache 布局已修正):`/tmp/opencode/rg2.py`/`rg3.py`。
- 服务器:serve-rg2(已做过一轮 profile,按规矩**重启后才能再压负载**)。
