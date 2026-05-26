# 配置（hyper-params）

> **共享契约** — 全部超参集中放在 config 里，便于调参。各功能文档引用本表对应行。

| 配置项 | 默认 | 含义 | 用于 |
| --- | --- | --- | --- |
| `update_planner.trigger_history_count` | 5 | `history_pool` 未处理数达到即自动触发 Update Planner | [04](04-update-planner.md) |
| `update_planner.max_revise_iterations` | 5 | LLM-1↔LLM-2 revise 轮数上限，超限强制定稿写入 candidate pool | [04](04-update-planner.md) |
| `benchmark_eval.min_samples` | 5 | 建议固化前所需的最小 `eval_runs` | [05](05-benchmark-evaluator.md) |
| `benchmark_eval.promote_threshold` | 10 | `positive` 达到即建议固化（在 ~10 个 task 最终成功代码里都出现） | [05](05-benchmark-evaluator.md) |
| `benchmark_eval.max_idle_evals` | 50 | candidate 经历这么多次 eval 仍 `used_runs==0` 且无人引用 → 建议遗弃（设大些，因为一整个 task set 可能都用不到某 candidate） | [05](05-benchmark-evaluator.md) |
| `benchmark_eval.schedule` | 夜间 cron | 固定一次；也可人工唤醒 | [05](05-benchmark-evaluator.md) / [06](06-heartbeat-cron.md) |
