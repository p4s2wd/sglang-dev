# sglang-codex — DeepSeek-V4-Flash 优化工作区

这个仓库是**工作产物**的版本控制:分析报告、A/B 实验脚本、测量数据、生产 launcher。
**sglang 框架源码不在这里**,它是独立的仓库(见下)。

## 三个目录的分工

| 路径 | 是什么 | 版本控制 |
|---|---|---|
| `/data/nvme/sglang-codex/sglang/` | sglang 框架源码,fork 自 sgl-project | ✅ 独立仓库,`github.com/p4s2wd/sglang-sm75` |
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
