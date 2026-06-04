# func_candidate_pool — 中期记忆（git-tracked）

> 自演化三层记忆里的**中期**一层。与短期、长期的区别在于**生命周期与是否进 git**：
>
> - **短期** `mem/history_pool/` —— 本地运行时态，**gitignored**，每个 checkout 各自一份。
> - **中期** `mem/func_candidate_pool/`（本目录）—— **进 git 跟踪**，候选函数 + 累积 stats
>   随仓库走，便于在**另一个集群上做大规模 simulation evaluation**（Benchmark Evaluator 在那边
>   持续累加 `*.stats.json`）。
> - **长期** `capx/skill_library/`、`capx/atomic_task_library/` —— 验证 + 人批准后固化的代码，走 PR。

## 内容（schema 见 `docs-se/storage.md`）

```
<func_name>.py          # 候选函数源码（Update Planner / LLM-2 写入）
<func_name>.stats.json  # 该候选的累积统计（Benchmark Evaluator 跨多次 eval 累加）
```

本 README 不是候选数据；工具按 `*.stats.json` / `<name>.py` 识别候选，不会把它当成函数。

## 为什么进 git

候选池是 dev 机产出、eval 集群消费的**共享中期产物**：dev 上 Update Planner 写候选 →
commit/push → eval 集群 pull 后在 sim 里跑 benchmark、累加 stats → 据阈值建议固化/遗弃（经人批准）。
让它随 git 流动，就不必另起一套同步通道。
