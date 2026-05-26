# Library Management — Benchmark Evaluator

> **状态**: 🔲 待实现
> **读 → 写**: `mem/func_candidate_pool/`（候选 + stats）→ 持续累加 stats；**经用户批准后**
> 固化到 `capx/skill_library/` 或 `capx/atomic_task_library/`，或从 pool 删除候选
> **依赖契约**: [storage.md](storage.md)（candidate stats schema、库路径）、
> [integration.md](integration.md)（注入点）、[config.md](config.md)（`benchmark_eval.*`）、
> [concepts.md](concepts.md)（VDM）

Update Planner 的产物是 `func_candidate_pool` 的更新；Benchmark Evaluator 是**短期记忆 →
长期记忆**的固化机制。

- **触发**：夜间 cron 固定一次（由 [06-heartbeat-cron.md](06-heartbeat-cron.md) 调度）；也可
  **人工唤醒**。仅当 `func_candidate_pool` 非空时执行。
- **评测**：类似 `scripts/run_agent0_qwen36_robosuite.sh` /
  `scripts/run_agent0_qwen36_libero.sh`，在模拟器里跑 benchmark 数据集，用 **VDM** 做自动
  成功判定。区别在于：`func_candidate_pool` 的内容会作为**可选工具**加入（注入方式见
  [integration.md](integration.md)），而不仅是已固化的 primitives / skill_library /
  atomic_task_library。
- **统计（每个 candidate，跨多次 eval 持续累积，跟随 candidate 直到其被删除；不记 negative）**：
  - `positive`：在最终成功代码里出现的累计 task 数（终版用到即 +1）。
  - `used_runs`：本轮 eval 中被调用过（chat 或终版任意处）则 +1。
  - `eval_runs`：candidate 进入 pool 后经历的 Benchmark Evaluator 次数，每轮 +1。
- **决策**：
  - **固化（promote）**：`positive ≥ benchmark_eval.promote_threshold`（默认 10，即在 ~10 个 task
    的最终成功代码里都出现）且 `eval_runs ≥ benchmark_eval.min_samples`（默认 5）→ 写入对应
    长期 library。被固化函数的 **docstring 要记更新历史日期**（不写具体改了什么）。
  - **遗弃（delete）**：candidate 在 pool 中经历 `eval_runs > benchmark_eval.max_idle_evals` 次
    评测**始终没被调用**（`used_runs == 0`）**且没有其他存活 candidate 引用它** → 从
    `func_candidate_pool` 删除。引用关系在删除前**动态扫描** pool 内其余 candidate 源码判定。
  - **保留（keep）**：其余情况——被用过但未达固化阈值、或仍被其他存活 candidate 引用 →
    暂留在 `func_candidate_pool`。
- **用户批准（强约束）**：Benchmark Evaluator 完成后生成一份**简单报告**，通知用户它认为
  哪些该固化、哪些该遗弃。**任何对长期 library 的 update / delete 都必须经用户批准；未经
  允许不得执行。**

> 设计取舍：我们**不记录 negative**（函数被试后弃用未必代表它差，可能只是该 task 不适用）。
> 固化只看 `positive` 阈值；遗弃只针对「长期占着 pool 又从没被用、且无人依赖」的 candidate；
> 最终都由用户把关。
