# 存储布局与数据 Schema（`mem/`）

> **共享契约** — 这是 pipeline 各阶段之间的数据接口：Feedback Postprocessor 写
> `history_pool`、Update Planner 读它写 `func_candidate_pool`、Benchmark Evaluator 读它并
> （经批准后）写长期 library。任何阶段都按这里的 schema 读写，互不依赖对方的实现。

所有 pool 都与 memory 相关，统一放在仓库的 `mem/` 目录下。

```
mem/
├── history_pool/                # 短期记忆：成功交互 log，openclaw 式追加，暂不删除
│   ├── <task>__<YYYYMMDD-HHMMSS>.json       # 原始成功 log（全文，drill-down 用）
│   └── <task>__<YYYYMMDD-HHMMSS>.digest.md  # distill 出的 sourced digest（消费者默认读这个）
├── func_candidate_pool/         # 候选函数（由 LLM-2 写入）+ 累积统计
│   ├── <func_name>.py           #   候选函数源码
│   └── <func_name>.stats.json   #   该函数的累积统计
└── .processed_history           # Library Management 已处理的 history id 列表(增量游标)
```

长期 library（固化目标，作为可 import 的代码模块，放在 `capx/` 包内；保持模块化但不过度
模块化）：

```
capx/skill_library/*.py          # 由 primitives 组成的通用函数
capx/atomic_task_library/*.py    # 每个文件 = atomic task 的函数 + 其 config(超参)
```

> 注意：候选**不直接改** `capx/skills/library.py`；本系统沉淀的新能力写入上面这两个**新模块**。
> 注入到 agent 可见工具的方式见 [integration.md](integration.md)。

## success log schema（`history_pool/*.json`）

> 由 [03-feedback-postprocessor.md](03-feedback-postprocessor.md) 写入；由
> [04-update-planner.md](04-update-planner.md) 消费。

```jsonc
{
  "task": "<task 名字>",
  "settings": { /* 用的设置：env / dataset / llm 等 */ },
  "final_code": "<Feedback Postprocessor 通用化后的最终代码>",
  "chat_history": [ /* 完整对话，含 human feedback */ ],
  "datetime": "YYYY-MM-DD HH:MM:SS"
}
```

## digest schema（`history_pool/*.digest.md`）

> 由 [03-feedback-postprocessor.md](03-feedback-postprocessor.md) 的 distill 步骤写入
> （Postprocessor 定稿后、入 pool 前）；由 [04-update-planner.md](04-update-planner.md)
> 的 LLM-1/LLM-2 **默认读取**（digest-by-default），需要核实时才 drill 回同名 `.json` 的
> `chat_history`。

设计目标：用尽量小的 context 承载"这条 history 值得沉淀什么"。借鉴 Agent Debugger 的
**Experience observability** —— 把大 trace 蒸成**小而可溯源**的 report。所以 digest **强制限长**
（`experience_distill.max_words`，见 [config.md](config.md)），且**每条论断都带回溯标注**
`[#<message_index>]`（指向同名 `.json` 里 `chat_history` 的 0-based 下标），便于 drill-down。

固定小节（Markdown）：

```md
# <task>  ·  <datetime>
- settings: <env / dataset / llm 等关键项>

## KEY STRATEGY
<final_code 的核心思路；为什么成功>  [#<idx>]

## REUSABLE PATTERN
<哪些步骤通用、可抽成 skill_library / atomic_task_library；建议粒度>  [#<idx>]

## KEY HYPER-PARAMS / FEEDBACK
<Postprocessor 超参化时锁定的值 + 其来历（哪条 human feedback 给的）>  [#<idx>]

## FRAGILITY
<看起来 fragile / 侥幸的地方；空则写 none>  [#<idx>]
```

> digest 是**有损**的入口视图，不是数据真相；真相永远在 `.json` 全文。任何要写进
> candidate 的论断，LLM-2 都应能 drill 回 `chat_history` 对应 `[#idx]` 核实。

## candidate stats schema（`func_candidate_pool/*.stats.json`）

> 由 [04-update-planner.md](04-update-planner.md) 创建；由
> [05-benchmark-evaluator.md](05-benchmark-evaluator.md) 持续累加并据此决策。

```jsonc
{
  "func_name": "<name>",
  "target_library": "skill_library | atomic_task_library",
  "positive": 0,                 // 在最终成功代码里出现的累计 task 数（≥promote_threshold 建议固化）
  "eval_runs": 0,                // 进入 pool 后经历的 Benchmark Evaluator 次数
  "used_runs": 0,                // 其中至少被调用过一次的 run 数（==0 且 eval_runs>max_idle_evals 触发遗弃判定）
  "source_history": ["<id>"],    // 来源 history
  "created_date": "YYYY-MM-DD",
  "last_eval_date": "YYYY-MM-DD"
}
```

> 引用关系（"是否有其他存活 candidate 依赖本函数"）在遗弃判定时**动态扫描** pool 内其余
> candidate 源码得出，不冗余存储。
