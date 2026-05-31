# Short-Term-GOAL —— 本次任务暂存

## 本次模块

④ Update Planner（`docs-se/04-update-planner.md`）。读 history_pool 增量 → 写 func_candidate_pool。
落地：`capx/self_evolve/{history_reader,proposal,update_planner}.py`，复用 ⓪ 的 MemStore/schemas。

> ③ 决议：核心已完成并单测；live-loop 接线 + 环境交互式 generalize 为 sim-gated，**暂挂**等真人在
> web/sim 联调时再做（用户拍板先做 ④）。③ 在 GOAL.md 仍 🔲，附注说明。

## 可执行步骤

- [ ] `history_reader.py`：只读工具集 `HistoryReader`（list_unprocessed / read_digest /
      read_history(field,offset,limit) / grep_history / read_library）。read_library 用 ast 解析
      capx/skill_library + atomic_task_library 的签名+docstring（目录不存在→空）。纯，可测。
- [ ] `proposal.py`：`ProposalCandidate`/`Proposal` schema + `validate_proposal`（机械护栏：
      source_history 引用真实 id、func_name 合法标识符、粒度启发式拒 put_apple 式绑物体、target 合法、
      与现有 candidate 去重）+ `write_accepted`（落 .py+.stats.json，含 source_history/created_date）。
- [ ] `update_planner.py`：tool-loop agent（解析/分发 tool 调用 + max_read_iterations 上限，
      用 scripted query_fn 可测）；LLM-1 propose / LLM-2 review 角色；`run_update_planner`
      编排 revise 上限 + 验证 gating + 持久化 + 推进 .processed_history 游标；`should_trigger`。
- [ ] `tests/test_update_planner.py`：fake agent/scripted query_fn 全程单测。
- [ ] 验证 pytest；__init__ 导出；commit。
- 唯一不可在此验证的：真 qwen 模型是否按 tool-call 约定输出（与 ③ sim 同性质，留 seam）。

## 进展 / 改了什么

- ⓪ 完成（d33ade6）。③ 核心完成（991d04a），接线暂挂。本轮做 ④。

## 还想改什么（收尾归纳）

- ③ 接线（sim-gated，见上）。⑤ Benchmark Evaluator 依赖 ④ 产出 + integration.md 注入点。
