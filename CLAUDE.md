# CLAUDE.md

> **每次开工第一步：读 `AGENT.md` 并完全按它走。** 本文件只做最薄的指针 + 不可违反的硬约束，
> 细节（ENV / 流程 / 风格 / 设计约束）一律以 `AGENT.md` 为准，别在这里重复。

## 开工流程（详见 `AGENT.md`）

1. 读 `AGENT.md`（开工须知）。
2. **问用户这次用哪个 task 探索**（当前主力：`open_drawer`）。
3. 进 `capx-se/<task>/`，先读其 `CONTINUE.md`（续接现状）再动手。
4. 按 `AGENT.md` 的硬性要求 coding（不读特权状态 / success+安全 / 不 overfit / 可多步 + wrist cam / 单文件<500 行）。
5. 验证通过就 `[tmp]` commit 落锚，并随手更新 `capx-se/<task>/CONTINUE.md`。

## 路标

- **开工须知**：`AGENT.md`
- **当前任务续接**：`capx-se/<task>/CONTINUE.md`
- **该任务发现的 cap-x 缺口**：`capx-se/<task>/GAPS.md`、`DISCUSSION.md`
- 历史背景（旧 self-evolve pipeline，已逐步过时）：`GOAL.md`、`docs-se/`、`Short-Term-GOAL.md`

## 不可违反的硬约束（细节见 `AGENT.md`）

- **解题路径不读 simulator 特权状态**：真机拿不到的（物体/关节真值）一律用 sensing 估计；特权读取只在 runner 测量。
- **success 必要但不充分，必须 SAFE**：不碰撞/扰动其它物体；安全优先于成功，不硬闯。
- **不 overfit 单 task/seed**：通用解法、闭环依赖感知反馈。
- **commit**：message 开头带 `[auto]`（旧的 `[tmp]` 历史 commit 不动）；不加 Co-Authored-By、不列 Claude 为 contributor。
- （eventual cap-x-se runtime 的设计北极星，仍参考）：success 只能由人给；更新长期 library 走 PR；术语 **atomic task library**；固化路径 `capx/skill_library/`、`capx/atomic_task_library/`。
