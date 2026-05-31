# Short-Term-GOAL —— 本次任务暂存

## 本次模块

③ Feedback Postprocessor + Experience Distill（`docs-se/03-*.md`）。
落地：`capx/self_evolve/feedback_postprocessor.py`，复用 ⓪ 的 `schemas.Digest` + `MemStore`。

设计要点：
- 流水线位置：live loop `human_finished`（async_trial_runner:674）之后、写 history_pool 之前。
- 两步串行：① generalize-by-rewrite（仍与环境交互，每轮只人判对错）→ ② Experience Distill（定稿后跑一次）。
- distill 固定四问 = digest schema 四小节；限长 `experience_distill.max_words`；每条带 `[#idx]` 溯源。

## 可执行步骤

- [x] 读 03-*.md + 01-*.md + storage/config + async_trial_runner 成功点 + query_model 契约。
- [ ] `feedback_postprocessor.py`：依赖注入式核心（`query_fn` / `judge_fn` 回调，不绑 web）：
      - 纯 prompt builders：`build_generalize_rewrite_prompt`、`build_distill_prompt`（含 indexed 转写）。
      - `parse_distill_response(text) -> Digest`（复用 schemas 四小节）。
      - `run_feedback_postprocessor(...)`：跑 generalize 多轮 + distill（限长 reshorten 守卫）+ 经 MemStore 写 history 对。
- [ ] `tests/test_feedback_postprocessor.py`：fake query/judge，验证 generalize 采纳/回退、distill round-trip、限长守卫、pool 落对。
- [ ] 验证：`.venv/bin/python -m pytest tests/test_feedback_postprocessor.py`。
- [ ] 收尾：把「live-loop 接线」列为 ③ 的下一步（需 web/sim 跑通才能验证，本轮不盲改 1155 行 runner）。

## 进展 / 改了什么

- ⓪ 已完成（branch se/storage-scaffold, d33ade6）。本轮在其上做 ③ 的可测核心。

## 还想改什么（收尾归纳）

- ③ 剩余：把 `run_feedback_postprocessor` 接进 async_trial_runner 的 `human_finished` 分支
  （generalize 复用现有 reset+regenerate+人判机制，只换 prompt 与「只判对错」语义），需起 web/sim 验证。
