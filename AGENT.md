# Coding Expectation —— 每日开工须知

> 每次开始任务先读这份。它告诉你：环境怎么用、去哪看进度、按什么风格和流程推进。

## 0. 我们在做什么（一句话）

在 cap-x agent0 之上做 **self-evolve**：把*人确认成功*的交互，经「通用化 → 蒸馏 → 候选 →
benchmark 自动验证 → 人批准」闭环，固化成**可验证、需人批准、可终身更新**的长期能力。
**设计真相在 `docs-se/`**（先读 `docs-se/README.md`）；**任务总账在 `GOAL.md`**。

## 1. ENV

- 整个 codebase 建立在 `.venv/bin/python` 环境上；部分依赖 `.venv-libero/bin/python`。
- web-ui 构建用 `~/.capx_nodeenv`。

## 2. 获知当前进度 → 选定本次任务

1. 读 **`GOAL.md`** —— 完整期待 + Pipeline 状态（哪些 ✅ / 哪些 🟡 / 哪些 🔲 / backlog）。挑出本次要推进的：
   优先 🔲（还没动的模块）；没有 🔲 时推进 **🟡 模块剩余的 live-gated 接线**（核心逻辑+单测已 done，
   只差接到真实 sim / web / gh 上跑通的那截）。
2. 读该模块对应的 **`docs-se/0X-*.md`**（先过一遍 `concepts.md` + `storage.md` 的术语与数据契约）。
3. 把这次的计划拆成可执行步骤写进 **`Short-Term-GOAL.md`**（临时 memory），分模块 coding。
4. 收尾 / 完成判据：
   - **🔲 模块**：核心逻辑写完 + 单测通过即可标 ✅。
   - **🟡 的 live-gated 接线**：没法纯单测，必须在真实环境里跑通一次才算完成 —— 例如
     ③ 真跑通一次「人确认成功」的 web trial 并落出成对 `.json`+`.digest.md`；⑤ 在有 `gh` 的环境真开出一个含
     promote/abandon 改动的 PR；⑥ cron 在设定时刻真触发一次 Evaluator。环境暂不具备时，在 `GOAL.md`
     备注里写清"卡在哪个外部依赖"，保持 🟡 而非误标 ✅。
   - `Short-Term-GOAL.md` 完成项随手标记；整块做完 → 清空它，并在 `GOAL.md` 对应行标 ✅。

## 3. 待实现系统（cap-x-se）的设计硬约束

> 这些是**我们要实现的 cap-x-se agent 在运行时必须满足**的流程/设计约束（来自 `docs-se`）。
> **不是**对"我在本仓库编辑、commit 代码"的限制 —— 本仓库照常正常编辑、正常 commit。

- **success signal 只能由人给**：cap-x-se 的 live loop 不让 VDM 判成功；VDM 只服务 Benchmark Evaluator 的自动评测。
- **更新长期 library 走 PR**：cap-x-se 与人在 manipulation task 交互后、要更新 library 时，**以 Pull Request 形式提交改动等人 merge**（而非弹聊天框确认）；它自己绝不直接改长期库 / 删候选。
- **遗弃不记 negative**；术语统一用 **atomic task library**（不绑具体物体/任务）。
- **固化路径**：长期库进 `capx/skill_library/`、`capx/atomic_task_library/`，**不**直接改 `capx/skills/library.py`；pool 在仓库根 `mem/`。

## 4. 代码风格

1. 结构清晰，不过度兜底。
2. 模块化又不过度模块化，单文件不超过 500 行。

## 5. 流程性要求

1. **自动修改**（直接动手，不用每步确认）。
2. **逐轮 debug 和验证**。
3. **每个验证通过的功能实现都 commit**：一旦某块功能改完、验证暂无明显 bug，就立刻 commit 落锚 ——
   这样中断（context 压缩 / session 切换）后也不会丢失追踪，能从最近一个 commit 干净续上。
   commit message **开头带 `[tmp]`**，便于区分我后续的改进与你维护的 commit；
   commit **不**加 Co-Authored-By、不把 Claude 列为 contributor。
4. **随手保持 `Short-Term-GOAL.md` 可接续**：每推进一步就把「这次改了什么 / 还想改什么」更新进去。
   我们拿不到 usage limit，靠 harness 在中断（context 压缩 / session 切换）后接着干 —— 只要
   `Short-Term-GOAL.md` 始终是最新的可执行进度，中断在哪都能无缝续上。
