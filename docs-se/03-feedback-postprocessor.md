# Feedback Postprocessor

> **状态**: 🔲 待实现
> **读 → 写**: 一次**人确认成功**的 trial（来自 [01-interactive-loop.md](01-interactive-loop.md)）
> → `mem/history_pool/<task>__<ts>.json` 的 `final_code`（通用化后的版本）
> **依赖契约**: [storage.md](storage.md)（success log schema）、[concepts.md](concepts.md)
> （atomic task / 超参化）

由于 human feedback 常带额外信息，我们在尝试时应**先尽量用这些信息让任务成功**。成功之后，
Feedback Postprocessor 负责把代码改得更通用：

- **流水线位置**：在「人判定成功」与「生成 final_code 写入 `history_pool`」**之间**。即进入
  短期记忆的已经是通用化后的版本。
- **做法**：多轮 code rewrite，去除对这次一次性额外信息的直接依赖。
  - 例：feedback 要求 eef 始终与 `z=0` 平面保持 `0.03` 距离。第一版可能直接在 main 里
    `z += 0.03`；rewrite 的更通用形态是——把这个 margin 变成 grasp 相关的超参，成为 grasp
    atomic task config 的一部分，通用于所有情况。
- **验证方式**：rewrite **仍需与环境交互**实际执行；**每轮只由人判对错**（不再接收细节
  feedback）。对 → 结束；错 → 继续重写。
- **输出**：通用化后的 `final_code` 连同完整对话（含 human feedback）按 success log schema
  追加到 `history_pool`（schema 见 [storage.md](storage.md)）。
