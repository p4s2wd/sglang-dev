# 长上下文复读循环 — 根因分析

分析对象:pi 通过 `http://api.mylab.io/v1` 调用 `deepseek-v4-flash` 时,回复反复输出
"Let me look at the MoE kernel and the overall pipeline. / Actually, let me reconsider. ..."。

会话证据:`~/.pi/agent/sessions/--data-nvme-sglang-codex--/2026-09-29T06-42-48-517Z_01a0ebe6-d444-702a-9c6a-12ff37fdd84a.jsonl`

## 结论

这是一个**真实的、可复现的语义闭环**,不是长度问题,也不是采样参数问题。
模型进入"我应该再确认一下"的吸引子后,在同一组结论之间无限循环,永远不发出工具调用。

## 复现

用会话最后 12 条消息(12K token)、greedy、带 tools,连续 3 次:

```
reconsider x34  comp=4000  finish=length  tool_calls=0
reconsider x34  comp=4000  finish=length  tool_calls=0
reconsider x34  comp=4000  finish=length  tool_calls=0
```

完整会话(104K token)greedy 复现同样特征:`finish=length`、16000/16000 token、
输出 67198 字符、`Actually, let me reconsider` **x219**。
真实会话里被中断的那条是 x119。

## 循环长什么样

```
The MoE kernel uses `mma.sync...`. So the path forward for attention is to
write a CUDA/PTX kernel.

But wait — let me reconsider whether there's a simpler optimization.
Let me look at the Triton source for the accelerate_matmul pass.
  → 找到的只是已知的 C++ pass,没有新信息
The MoE kernel uses hand-written PTX mma.sync. So the path forward for
attention is to write a CUDA/PTX kernel.   ← 回到起点
Actually, let me reconsider.
```

出现次数最多的句子(一次失败运行内):

| 次数 | 句子 |
|---|---|
| x26 | Actually, let me reconsider. |
| x10 | So the path forward for attention is to write a CUDA/PTX kernel. |
| x9 | Let me look at the Triton source for the accelerate_matmul pass. |
| x9 | The MoE kernel uses hand-written PTX mma.sync. |

**失败是二元的**:要么 157 token / 10 秒内给出正确工具调用,要么烧完整预算且不调工具。

## 已排除的假设(都有数据)

| 假设 | 结论 | 依据 |
|---|---|---|
| 长上下文导致 | **排除** | 12K token 就循环(x34),比 104K 那档更严重 |
| fp8_e4m3 KV cache 精度损失 | **排除** | 换 bf16 KV 后同 prompt 输出**逐字节相同** |
| greedy 贪心解码 | **是机制,但非唯一可控因素** | 见下表 |
| 服务端工具解析器漏解析 | **排除** | `DeepSeekV4Detector` 单测能正确解析;不带 tools 时模型才吐裸 DSML,那是探针问题 |
| 会话里那条 aborted 消息是元凶 | **排除** | 带上/不带它回放都不循环 |
| 网关篡改 temperature | **排除** | temp=0 时 5 次输出 1 唯一,temp=0.6 时 5 唯一,直连网关一致 |

## 采样参数:只是挪动概率,不能消除

同一 prompt,每档 3 次(104K 上下文)。**单次对比会得出相反结论,必须重复** ——
本次调查中单次运行曾把"温度即可修复"和"温度反而更糟"互相推翻:

| 配置 | `Actually, let me reconsider` | 循环/总次数 |
|---|---|---|
| greedy | 4, 0, 0 | 1/3 |
| temperature 0.6 | 17, 0, 12 | 2/3 |
| temp 0.6 + repetition_penalty 1.1 | 0, 0, 0 | 0/3 |
| 同上,追加 2 次 | 4, 9 | 2/2 → 共 2/5 |

`repetition_penalty 1.1` 单独配 greedy 几乎无效(x19 → x18),因为重复的短语太短。

## 加重因素:上下文里已有的复读文本

最后 12 条消息里就带着会话早期累积的复读句。删掉它们、长度几乎不变(36601 → 34305 字符):

| | `reconsider` | 结果 |
|---|---|---|
| 原样 | x34 | 循环 |
| 删掉复读句 | x16 | 仍循环,减半 |

即复读文本**在上下文中自我强化**,但删掉它并不能消除闭环。

## pi 侧的放大机制

```
// pi: Skip aborted messages with no content
if (msg.stopReason === "aborted" && msg.content.length === 0) return false;
```

只有**无内容**的 abort 消息会被丢弃。会话中最后那条:

```
stopReason: "aborted", errorMessage: "Operation aborted", usage 全 0,
无 text、无 toolCall,只有 21855 字符复读 thinking
```

它有内容,所以留在历史里,下一轮模型看到自己复读的文本,继续复读。

`maxTokens: 16000` 让每次循环烧掉 16000 token(约 2–5 分钟)才被发现。

## 网关确实指向本机

`api.mylab.io` = 192.168.2.2(New API 网关),本机 = 192.168.2.6。
经网关发一个带 BOS 的请求,本机服务日志的 `Stripped <|begin_of_sentence|>` warning
从 95 涨到 96,且直连与经网关的 usage / 输出文本逐字相同 —— **网关后面就是这台机器的 8200**。

## 第二个案例:opencode 重复调用同一条 grep(更干净,已完整复现)

session `ses_f12a25898ffe1ypJUExVYX9aQm`("项目进度读取与报告"),
provider `myai-sh` → `http://api.mylab.io/v1` → **同一台机器的 8200**,同样未配采样参数(贪心)。

### 现场数据

连续 **34 轮**,每一条的**工具调用、工具输出(3342 字符)、输出 token 数(364)完全相同**:

```
cd /data/nvme/sglang-codex && grep -rn "decode\|tok/s\|13.55\|11.00\|50\|bottleneck\|瓶颈\|W4A16\|12 GB/s\|9.4ms" sglang-sm75-progress.md ... | head -20
```

grep 的输出每次都是同样的 3342 个字节 —— **模型没有任何新信息,却一直重复同一条命令**。
注意这次 `reasoning` token 全程为 0(provider 未开思考),所以它是**纯工具调用循环**,
和 pi 的"思考闭环"是两种不同的失效,但都不可自愈。

锁定的形成过程(比想象的更早):

| idx | input tokens | out | 说明 |
|---|---|---|---|
| 16–27 | 49000 → 71500 | 338–364 | 已 8 次跑几乎相同的 grep(尚未逐字节相同) |
| 28 起 | 73526 → 152292 | 364 | 逐字节相同,持续 34 轮直到用户 abort |

`MessageAbortedError` 出现在 3 个位置,全部是用户手动中断。

### 复现(确定性,greedy)

用 DB 里的真实会话重建到 idx 50,补上工具结果:

| 配置 | 复现同一条 grep |
|---|---|
| greedy(线上配置) | 2/2 |
| temperature 0.6 | **2/2** |
| 加显式 system 规则("不要重跑已有输出的命令") | **4/4** |
| greedy + repetition_penalty 1.05 | 逐字节相同 |

**在模型层面什么都无效。** 温度、惩罚、明确的系统指令,全部无效。

### 唯一有效的干预:客户端拦截

把重复的那次工具结果替换成提示,而不是再返回一遍同样的字节:

```
NOTE: this exact command was already run earlier in this conversation and its
full output is already present above. Re-running it returns identical bytes.
Do not repeat it. Choose a different file, a different pattern, or answer now.
```

结果:模型**换了目标文件**(`sglang-sm75-progress.md` → `sglang-sm75-runbook.md`),
输出从 363 token 降到 186,2/2 确定生效。

## 定性:是模型的问题(已排除我方)

针对"会不会是服务端的锅"的质疑,逐条验证:

### 1. 往返保真 —— 不是历史被污染

模型发出的 DSML → `DeepSeekV4Detector` 解析 → OpenAI `tool_calls` →
`encoding_dsv4.encode_messages` 回渲,4/4 全部一致:模型看到的就是它自己写的东西,
命令一字不差,工具结果也在。没有二次编码、没有丢字段。

> 注意:该检测器把 name 和 arguments 分两条 `ToolCallItem` 返回
> (一条 `name='bash'` 参数为空,一条 `name=None` 带参数 JSON),
> 只取第一条会得到空参数 —— 这是排查时踩到的坑。

### 2. 基本能力正常 —— 模型会调工具

短上下文下正确发起 `ls -la /tmp/*.md`;真实 session 里连续 32 轮 `stopReason=toolUse` 成功。
往返格式没问题。

### 3. 决定性实验:唯一的变量是「对话结尾是什么」

同一段历史(N 次完全相同的 命令+结果),只改最后一条:

| 结尾 | 6 次重复 | 20 次重复 |
|---|---|---|
| user 消息("现在总结一下") | **0/2** 重跑 | **0/2** 重跑 |
| **工具结果**(让模型继续) | **2/2** 重跑 | **2/2** 重跑 |

**986 token 就 2/2 逐字节重跑。** 不需要长上下文。

### 4. 上下文长度不是变量

用真实 session 的最后 K 条历史(该命令已出现过许多次):

| K | 8 | 16 | 24 | 32 | 48 | 64 | 97 |
|---|---|---|---|---|---|---|---|
| input tokens | 8427 | 16535 | 24643 | 32751 | 48007 | 66685 | 110597 |
| 重跑 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |

K=8 时 8427 token 就已经 2/2 了。

### 结论

模型在**「刚拿到工具结果、必须决定下一步」**这个位置有模式续写缺陷:
它会重发最近一次工具调用,而不是往前走。只要没有新的用户消息打断,这个模式就会被不断强化
——线上那 34 轮就是这么滚起来的。

模型**并非不可救**:显式告诉它"这条你刚跑过"之后,它确实换了目标文件(见上)。
也就是说它**能被说服,但不会自己想起来**。所以唯一有效的防线在客户端。

### 我方无责的补充证据

`encoding_dsv4.py` 是手写的 v4 编码器,仓库里没有 dsv4 的参考 jinja 可比对
(只有 v31/v32/v3)。但从行为看格式是对的:模型连续 32 轮正确调工具、
assistant 的 DSML 原样回显。若格式错了,不会是这个表现。

## 建议(按性价比)

1. **别把 1M 当作可用窗口。** 实测 12K 就会循环,宣称的 1M 显然不可依赖。
   给 pi 配一个保守的 `contextWindow`,让它在退化前先压缩历史。
2. **切断自我强化。** 让 pi 丢弃带内容的 aborted 消息(改上游 pi,或本地 patch)。
   代价最小、收益明确。
3. **加 `samplingParams`。** pi 的 models.json 支持任意采样参数透传,且实测网关原样转发。
   `{"temperature": 0.6, "repetition_penalty": 1.1}` 只能降低概率,不能保证。
4. **考虑服务端兜底。** 若要根治,方向是 n-gram 循环检测 + 提前截断
   (SGLang 目前没有内建机制),而不是调采样参数。

## 复现脚本

| 脚本 | 用途 |
|---|---|
| `depth_threshold.py` | 上下文深度扫描 |
| `loop_mitigation_ab.py` | 采样参数 A/B |
| `abort_poison_repro.py` | 忠实回放会话(含 thinking 内联) |
| `longctx_ab.py` | 长上下文 A/B 仪表 |

## 方法论教训(本次调查)

1. **单次 A/B 在随机失效上不可信。** 本次至少有两次结论被重复实验推翻。
2. **注意 radix prefix cache**:同样的 greedy 请求,先跑和后跑可能不同
   (缓存命中导致分块边界与 fp8 KV 数值不同)。深度扫描里 36/71 档三次结果不同,12/20/28 档三次相同。
3. **别只读 `content`**:reasoning parser 会把思考放进 `reasoning_content`,
   预算小的时候 `content` 为空但回复其实正常 —— 本次因此误判过一次"空输出"。
4. **要带 `tools`**:不带 tools 时模型会吐裸 DSML 文本,看起来像工具解析 bug,其实是探针问题。
5. **预算要够**:这个模型每轮思考 60–200 reasoning token,`max_tokens=200` 会让健康回复看起来像彻底失败。
