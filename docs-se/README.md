# Self-evolvable cap-x — 设计文档

本目录描述 cap-x 下一代 self-evolve agent 系统的设计。设计借鉴 openclaw
（参考 `/leonardo/home/userexternal/zwang003/Projects/openclaw`）的 memory /
heartbeat / cron 机制，目标是：在拿到 human feedback 之后，agent 有一套**可自我验证、
需人类批准、可终身更新**的流程来沉淀新能力。

文档按**功能拆分**：互不影响的功能各占一个文件，可由独立的 agent 分别实现。共享的
"契约" 文档（术语、存储 schema、注入点、配置）被多个功能引用——改它们会影响多个功能，
所以单独抽出、谨慎修改。

---

## 1. 起点：cap-x agent0

我们的设计基于 **cap-x agent0**。它已具备：

- 基本的环境交互能力；
- 基本的 planning 能力；
- **VDM**（cap-x 内的一个 agent，用于自动评测 task progress / 判定成功失败）。

agent0 的能力分层：

- **primitives**：一组与环境交互的 API（最底层）。
- **skill library**：由 primitives 组合而成的通用函数。在 agent0 语境下**不涉及
  subtask**。

我们还实现了与人类的远端交互通信：可以选择数据集，默认 LLM 为服务器上的
`qwen3.6-27B`。

self-evolve 系统在 agent0 之上**新增 / 部分重新设计**，使其能把交互中得到的成功经验，
经过验证后固化为长期能力。

---

## 2. 端到端 Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ [Live Loop]  human-in-the-loop 执行任务 (主交互循环)                    │   01-interactive-loop.md ✅
│   人判定 success  ← 唯一的 success signal，必须人给                      │   (可视化见 02-visualization.md ✅)
└───────────────┬─────────────────────────────────────────────────────┘
                │ 若本次依赖了 human feedback 的额外信息才成功
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [Feedback Postprocessor]                                              │   03-feedback-postprocessor.md 🔲
│   多轮 rewrite（仍与环境交互），把代码通用化 / 超参化                     │
│   每轮只由人判对错（不再接收细节 feedback）：对→结束，错→重写             │
└───────────────┬─────────────────────────────────────────────────────┘
                │ final_code
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [Experience Distill]  postprocessor 定稿后 / 入 pool 前                │   03-feedback-postprocessor.md 🔲
│   debugger 式问题驱动 → 把整段 trial 蒸成小而可溯源的 digest（限长）     │
└───────────────┬─────────────────────────────────────────────────────┘
                │ final_code + digest 一起入 pool
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ mem/history_pool/  <id>.json(全文,drill-down) + <id>.digest.md(默认读) │   storage.md
└───────────────┬─────────────────────────────────────────────────────┘
                │ 触发：未处理 history 数 ≥ trigger_history_count（默认 5）
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [Update Planner]  (Library Management 的一部分)                        │   04-update-planner.md 🔲
│   LLM-1: 默认读 digest / 按需 drill 原文(Agent Debugger 式)→ 提议代码   │
│   LLM-2: 同式回查审核(去重 / bug / general / 粒度 ≤ atomic) ──revise─┐ │
│          revise 上限 = max_revise_iterations；超限则强制定稿         │  │
│          └──pass──► 由 LLM-2 写入 func_candidate_pool ◄──────────────┘  │
│   完成后：标记这批 history 为已处理（history 本身保留）                  │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ mem/func_candidate_pool/   候选函数 + 累积统计（统计跟随 candidate）     │   storage.md
└───────────────┬─────────────────────────────────────────────────────┘
                │ 触发：夜间 cron 固定一次（也可人工唤醒）；pool 非空时执行
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [Benchmark Evaluator]  (短期→长期记忆的机制；用 VDM 自动评测)            │   05-benchmark-evaluator.md 🔲
│   把 candidate 注入"可选工具"，在 sim 里跑 benchmark                    │
│   累积统计 → 生成报告 → ★ 必须用户批准；未经允许不得 update             │
└───────────────┬─────────────────────────────────────────────────────┘
                │ 用户批准后 → 固化 / 遗弃 / 保留
                ▼
   capx/skill_library/ · capx/atomic_task_library/   (长期记忆)              storage.md

调度（贯穿全程）: [Heartbeat / Cron] 唤起 daily task + 定时触发 Evaluator       06-heartbeat-cron.md 🔲
```

---

## 3. 文档地图

**共享契约**（被多个功能引用，改动需谨慎）：

| 文档 | 内容 |
| --- | --- |
| [concepts.md](concepts.md) | 术语表（全文术语必须一致） |
| [storage.md](storage.md) | `mem/` 存储布局 + 数据 schema + 长期库路径（pipeline 各阶段的数据契约） |
| [integration.md](integration.md) | 与现有代码对接：`SkillLibrary` 注入点 |
| [config.md](config.md) | 全部 hyper-params 速查表 |

**功能模块**（互不影响，可分别实现；✅ 已实现 / 🔲 待实现）：

| 文档 | 状态 | 读 → 写 |
| --- | --- | --- |
| [01-interactive-loop.md](01-interactive-loop.md) | ✅ | human → 成功 trial（喂给 Feedback Postprocessor） |
| [02-visualization.md](02-visualization.md) | ✅ | 观测帧 → viser 回放 + `outputs/.../attempt_NN/observations.npz`（纯 UI，不影响 pipeline 数据） |
| [03-feedback-postprocessor.md](03-feedback-postprocessor.md) | 🔲 | 成功 trial → `history_pool`（通用化 `final_code` + 入 pool 前 distill 出 `.digest.md`） |
| [04-update-planner.md](04-update-planner.md) | 🔲 | `history_pool`（默认读 digest / 按需 drill 原文）→ `func_candidate_pool` |
| [05-benchmark-evaluator.md](05-benchmark-evaluator.md) | 🔲 | `func_candidate_pool` →（批准后）长期 library |
| [06-heartbeat-cron.md](06-heartbeat-cron.md) | 🔲 | 调度：唤起 daily task + 定时触发 Evaluator |

**建议阅读顺序（实现某功能时）**：先读 [concepts.md](concepts.md) + [storage.md](storage.md)
（弄清术语与数据契约），再读你要实现的那个功能文档；只有 Benchmark Evaluator 额外需要
[integration.md](integration.md)。各功能文档里用到的配置项都指向 [config.md](config.md)。

---

## 4. 全局硬约束（实现任何功能都不得违反）

这些是用户拍板的、容易在实现中被忽略的约束；每条的细节在对应功能文档里：

- **success signal 只能由人给**：VDM 不在 live loop 判定成功；VDM 仅用于 Benchmark
  Evaluator 的自动评测。（见 [01-interactive-loop.md](01-interactive-loop.md)）
- **任何对长期 library 的 update/delete 必须经用户批准**：Benchmark Evaluator 只生成报告
  + 等批准。（见 [05-benchmark-evaluator.md](05-benchmark-evaluator.md)）
- **遗弃不记 negative**：只按 `positive` 阈值固化、按"久占 pool 又没人用且无人引用"遗弃。
  （见 [05-benchmark-evaluator.md](05-benchmark-evaluator.md)）
- **术语统一**：用 "atomic task library"（旧称 "subtask library" 已废弃）；atomic task
  不绑具体物体/任务。（见 [concepts.md](concepts.md)）
- **路径**：长期库固化到 `capx/skill_library/`、`capx/atomic_task_library/`，**不**直接改
  `capx/skills/library.py`；pool 在仓库根 `mem/`。（见 [storage.md](storage.md)）
- **Memory Compact 暂不做**；rgbd video feedback 为未来。（见 [concepts.md](concepts.md)）
