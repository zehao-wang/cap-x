# Library Management — Update Planner

> **状态**: 🔲 待实现
> **读 → 写**: `mem/history_pool/`（未处理增量）+ `mem/.processed_history`（游标）
> → `mem/func_candidate_pool/`（`<func>.py` + `<func>.stats.json`）
> **依赖契约**: [storage.md](storage.md)（history schema、candidate stats schema、库路径）、
> [concepts.md](concepts.md)（atomic task 粒度规则）、[config.md](config.md)
> （`update_planner.*`）

Library Management 管理并更新 `atomic_task_library` 与 `skill_library`。Update Planner 是其
**前半段**（后半段 = [05-benchmark-evaluator.md](05-benchmark-evaluator.md)）。

- **触发**：当 `history_pool` 中**未处理**的 history 数达到 `update_planner.trigger_history_count`
  （默认 5）自动开始（比 Benchmark Evaluator 更频繁）。
- **不能无限全读**：必须记录已处理过哪些 history（`mem/.processed_history`，见
  [storage.md](storage.md)），每次只处理增量。history 本身保留、暂不删除。
- **流程**：
  1. **LLM-1**：input = 这批未处理 history 及其对应 task；output = 一些应新增 / 更新的代码。
  2. **LLM-2**：基于 LLM-1 的产出，结合**当前 library** 审核：是否重复功能、是否引入 bug、
     是否够 general、粒度是否超出 atomic task（拒绝 `put_apple()` 这类绑定具体物体/任务的
     函数——粒度规则见 [concepts.md](concepts.md)）。
     - pass → **由 LLM-2 把函数写入 `func_candidate_pool`**（同时创建 `*.stats.json`，schema 见
       [storage.md](storage.md)）。
     - 不 pass → 带着 feedback 唤醒 LLM-1 去 revise。
  3. **终止**：revise 轮数上限 = `update_planner.max_revise_iterations`（默认 5）。几轮 feedback
     + 查错后若仍未收敛，**强制要求模型定稿一个方案写入 `func_candidate_pool`**。
  4. 处理完这批后，标记其为已处理；这批 history 的对话已无进一步意义（但文件保留）。
