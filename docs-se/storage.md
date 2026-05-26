# 存储布局与数据 Schema（`mem/`）

> **共享契约** — 这是 pipeline 各阶段之间的数据接口：Feedback Postprocessor 写
> `history_pool`、Update Planner 读它写 `func_candidate_pool`、Benchmark Evaluator 读它并
> （经批准后）写长期 library。任何阶段都按这里的 schema 读写，互不依赖对方的实现。

所有 pool 都与 memory 相关，统一放在仓库的 `mem/` 目录下。

```
mem/
├── history_pool/                # 短期记忆：成功交互 log，openclaw 式追加，暂不删除
│   └── <task>__<YYYYMMDD-HHMMSS>.json
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
