# Heartbeat / Cron：长期目标与 Daily Task

> **状态**: 🔲 待实现
> **角色**: 调度层，贯穿全程——唤起 daily task（产生新成功 history）并定时触发
> [05-benchmark-evaluator.md](05-benchmark-evaluator.md)
> **依赖契约**: [concepts.md](concepts.md)（Heartbeat / Cron 定义）、[config.md](config.md)
> （`benchmark_eval.schedule`）

- **长期目标**：① 提升在 benchmark 上的效果；② 在人机交互的 task 上更快、更容易成功。
- **Daily task（由 heartbeat 唤起）**：把之前**正确率低**的 task 拎出来交给人 feedback
  （走 [01-interactive-loop.md](01-interactive-loop.md)），通过新产生的成功 history 推动
  library 更新。
- **Cron**：在固定时刻触发 heartbeat。典型用法——夜间固定时间触发 Benchmark Evaluator 做
  长期记忆更新（同时 Update Planner 由 history 增量阈值自行更频繁触发，见
  [04-update-planner.md](04-update-planner.md)）。
