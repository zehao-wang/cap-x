# Feedback Postprocessor

> **状态**: 🔲 待实现
> **读 → 写**: 一次**人确认成功**的 trial，经 **postprocess handoff**（来自
> [01-interactive-loop.md](01-interactive-loop.md)）读入
> → `mem/history_pool/<task>__<ts>.json`（通用化后的 `final_code` + 全文）
> **和** 同名 `<task>__<ts>.digest.md`（distill 出的 sourced digest）
> **依赖契约**: [storage.md](storage.md)（**postprocess handoff schema** + success log schema +
> digest schema）、[concepts.md](concepts.md)（atomic task / 超参化）

## 输入契约：postprocess handoff（live-loop 落盘）

本模块**唯一的输入**是 live loop 在人点 Finish 确认成功时写到 trial 根目录的
`postprocess_handoff.json`（schema 见 [storage.md](storage.md)），**不**再去反解 per-attempt 的
trace。读取/校验走 `capx.self_evolve.handoff.load_handoff()`，它把 handoff 映射到本模块入口
`run_feedback_postprocessor(...)` 的关键字参数：

- `original_code` ← handoff `final_code`（人确认成功的码，rewrite 的安全基线）
- `task` / `task_description` ← handoff `task`
- `settings` ← handoff `settings`
- `chat_history` ← handoff `chat_history`（含 verbatim 人 feedback 作为一等 turn，供下面 distill 的
  `[#idx]` 直接溯源）

> 实现态：core 已是依赖注入式（`query_fn` / `judge_fn` / `store`），rewrite 仍需**与环境交互**由
> `judge_fn` 落到真 sim/真机；离线只想验证「读 log + distill agent 产出」可用
> `python -m capx.self_evolve.debug postprocessor <trial>`（见 [debugging.md](debugging.md)）。

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

## Experience Distill（定稿后、入 pool 前）

> 独立于上面的 rewrite：rewrite 只负责把**代码**通用化；distill 负责把**这整段经验**蒸成一份
> 小而可溯源的 digest。两步串行——rewrite 定稿 `final_code` 之后、把 trial 写进 `history_pool`
> **之前**，跑一次 distill，让 `.json`（全文）和 `.digest.md`（digest）一起入 pool。

**为什么放这里**：这是借鉴 Agent Debugger 的 **Experience observability**——下游 Update Planner
若把每条 history 的 `chat_history` 全文灌进 context 会爆窗口；正解是**先蒸成小 digest，消费者
默认读 digest、需要时再 drill 回全文**。把 distill 放在入 pool 前，大 context 的代价**只在写入
时一次性付掉**，之后所有消费者都在小 digest 上工作。

- **做法**：问题驱动的 debugger 式分析（不是泛泛"总结一下"），针对成功 trial 问固定问题：
  KEY STRATEGY / REUSABLE PATTERN（可抽成哪个库、什么粒度）/ KEY HYPER-PARAMS+来历 / FRAGILITY。
  此刻整段 trial 仍在 Postprocessor 的 context 里，distill 近乎零额外读取成本。
- **强制限长**：`experience_distill.max_words`（默认 200，见 [config.md](config.md)），逼出"小"。
- **强制 sourcing**：每条论断带 `[#<message_index>]` 回溯到 `chat_history` 下标——digest 是有损
  入口视图，全文才是真相，下游靠这些标注 drill-down 核实。
- **输出**：`<task>__<ts>.digest.md`，schema 见 [storage.md](storage.md) 的 digest schema。
