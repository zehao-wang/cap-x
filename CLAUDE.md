# CLAUDE.md

> **每次开工第一步：读 `AGENT.md` 并完全按它走。** 本文件只做最薄的指针 + 不可违反的硬约束，
> 细节（ENV / 流程 / 风格 / 设计约束）一律以 `AGENT.md` 为准，别在这里重复。

## 开工流程（详见 `AGENT.md`）

1. 读 `AGENT.md`（开工须知）。
2. 读 `GOAL.md` → 看「Pipeline 状态」挑本次要推进的 🔲 模块。
3. 读该模块对应的 `docs-se/0X-*.md`（先过 `concepts.md` + `storage.md` 的术语 + 数据契约）。
4. 把本次计划拆成步骤写进 `Short-Term-GOAL.md`（临时暂存），分模块 coding。
5. 模块做完 → 在 `GOAL.md` 对应行标 ✅、清空 `Short-Term-GOAL.md`。

## 路标

- **开工须知**：`AGENT.md`
- **进度总账 / 任务分解**：`GOAL.md`
- **设计真相**：`docs-se/`（先 `README.md`）
- **本次任务暂存**：`Short-Term-GOAL.md`
- **暂缓候选**：`TASK_TMP_ASIDE.md`

## 不可违反的硬约束（细节见 `AGENT.md` §3 / `GOAL.md` §1）

- **success 只能由人给**：cap-x-se 的 live loop 不让 VDM 判成功。
- **更新长期 library 走 PR**：等用户 merge（批准）/ close（否决），绝不直接改长期库 / 删候选。
- **遗弃不记 negative**；术语统一 **atomic task library**。
- **固化路径**：`capx/skill_library/`、`capx/atomic_task_library/`；不直接改 `capx/skills/library.py`；pool 在仓库根 `mem/`。
- **commit**：message 开头带 `[tmp]`；不加 Co-Authored-By、不列 Claude 为 contributor。
