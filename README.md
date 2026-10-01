# sglang-codex — DeepSeek-V4-Flash 优化工作区

这个仓库是**工作产物**的版本控制:分析报告、A/B 实验脚本、测量数据、生产 launcher。
**sglang 框架源码不在这里**,它是独立的仓库(见下)。

---

## 🔴 生产红线:不要在生产代码目录里切分支

**2026-10-02 事故**:为发布 `sm75main2` 在 `/data/nvme/sglang-codex/sglang/`
里执行了 `git checkout -b release/sm75main2`。那条分支基于 `release/sm75main1`
——9/30 上午的快照,不含当天晚上 7 个 commit 里的 `sglang/kernels/ops/moe/moe_align.py`。
该模块是**懒加载**(首次跑 MoE 才 import),所以服务启动正常、**一提问就崩**:

```
ModuleNotFoundError: No module named 'sglang.kernels.ops.moe.moe_align'
```

级联的 `gloo ... Connection closed by peer` 只是噪声。**不是 IMA 回归。**

### 硬规则

| 规则 | 说明 |
|---|---|
| **不在 `/data/nvme/sglang-codex/sglang/` 切分支** | 这是生产实际加载的代码(editable install 指向它)。切分支 = 换掉整棵源码树 |
| **不在 `/data/nvme/sglang/` 改任何东西** | 生产运行目录:launcher、venv、release bundle。不是 git 仓库 |
| **要发版/看别的分支,用 `git worktree` 或 `git archive`** | 在**另一个目录**里操作,生产工作树一个字节都不动 |
| **动完生产代码目录,必须验证服务还活着** | `pgrep -f '[b]in/sglang'` + `curl -s localhost:8200/health` |

### 每次开工先确认生产状态

```sh
cd /data/nvme/sglang-codex/sglang
git branch --show-current      # 必须是 sm75-dsv4-flash-main
git rev-parse --short HEAD      # 应为 331faaeaf7
grep -c "page_ids < page_table_width" python/sglang/kernels/ops/attention/dsv4/topk.py   # 应为 1
ls python/sglang/kernels/ops/moe/moe_align.py                                          # 应存在
```

### 概念上要分清的两件事

- `release/sm75mainN` **分支** = 发版快照,含 wheel 等产物,**它的源码树 ≠ 开发线**。
- `sm75-dsv4-flash-main` = 开发线,生产跑的就是它。

在同一个目录里对两者做 checkout 就会踩到本次事故。

---

## 三个目录的分工

| 路径 | 是什么 | 版本控制 |
|---|---|---|
| `/data/nvme/sglang-codex/sglang/` | **生产实际加载的代码**,sglang 框架源码,fork 自 sgl-project | ✅ 独立仓库,`github.com/p4s2wd/sglang-sm75`,生产分支 `sm75-dsv4-flash-main` |
| `/data/nvme/sglang-codex/`(本仓库) | 分析报告、实验脚本、测量数据、生产 launcher 副本 | ✅ 本仓库 |
| `/data/nvme/sglang/` | 生产运行目录,真正被执行的 launcher 在此 | ❌ 不是仓库,见下方「launcher 同步」 |

**框架改动走 sglang 仓库**(例如 `dsv4/topk.py` 的 IMA 修复),
**实验与配置走本仓库**。两者不要混:框架源码树里不应该出现某个私有集群的启动脚本。

## 重要文件

| 文件 | 内容 |
|---|---|
| `plan2026-09-30/NOTES.md` | **主文档**。§0 结论速览、§6.5–§6.7 IMA 破案全过程、§6.8 prefill graph 证否、§6.9 F3 层重排、§6.10 EAGLE 结案 |
| `plan2026-09-30/res/` | 上面每个数字的原始测量数据(`*.json` 结果 + `*.log` 服务日志) |
| `plan2026-09-30/{ab_restart,measure,ab_compare}.py\|sh` | A/B 方法学工具链:重启换 flag → 压负载 → 交错比对 |
| `launcher/deepseek-v4-flash.sh` | 生产 launcher 的受控副本(见下) |
| `PREFILL_PROFILE.md` | prefill 侧历史结论(含 §"负载均衡" = F3 的原始 +8.2% 记录) |
| `PROGRESS.md` | 全局进度 |

## launcher 同步

真正被执行的是 `/data/nvme/sglang/deepseek-v4-flash.sh`,**不在任何仓库里**。
本仓库的 `launcher/deepseek-v4-flash.sh` 是它的受控副本,提交前请核对:

```sh
diff launcher/deepseek-v4-flash.sh /data/nvme/sglang/deepseek-v4-flash.sh
```

改了生产文件后记得 `cp` 一份回来提交。当前关键默认值:

- `SGLANG_PP_LAYER_PARTITION=11,11,11,10` —— F3 层重排,**prefill +7.6%**
  (NOTES §6.9)。回滚:`SGLANG_PP_LAYER_PARTITION= ./deepseek-v4-flash.sh`
  ——注意用 `-` 而不是 `:-` 的写法正是为了让显式空值能穿透。
- `PREFILL_GRAPH=disabled` —— prefill CUDA graph 实测慢 2.9 倍,不要打开(NOTES §6.8)
- `SGLANG_PP_EARLY_PROXY_SEND=0` —— 实测 −30%,保持 0(NOTES §4)
- `SPEC_ALGO=` —— EAGLE 算术上装不下,保持空(NOTES §6.10)

## 不跟踪的东西

见 `.gitignore`。三个大目录:`sglang/`(独立仓库)、`.venv/`(8.5 GB)、
`profiles/`(1.3 GB trace)。另外 `plan2026-09-30/res/coredumps/` 有 382 MB
coredump 被排除 —— NOTES §6.7 记录了原因:文件写完整但 `e_phnum = 0`,
所有 gdb 都拒绝读,留字节只是暗示它还值得调试。
