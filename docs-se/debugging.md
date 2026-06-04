# 模块化 Debug 接口（develop 阶段）

> **共享工具** — 最终系统是端到端自动的；develop 阶段我们要能**逐模块、孤立地**驱动 pipeline，
> 并**看见**每个模块读到了什么、产出了什么。
> **实现**：`capx/self_evolve/debug.py`（CLI + 可注入的 `query_fn`/`judge_fn`）+
> `capx/self_evolve/handoff.py`（live-loop → postprocessor 的输入契约读取）。

## 设计原则

pipeline 每个模块都通过**注入的回调**接触模型与环境（`query_fn(messages)->str`、`judge_fn(code)->bool`、
`store: MemStore`）。所以 debug 不重写模块逻辑，只提供**可观测**版本的回调 + 一层薄 CLI 把
「输入 / 中间 LLM 调用 / 输出」打印出来。每个 debug 关注点对应一个子命令：

```
python -m capx.self_evolve.debug <handoff | postprocessor | planner> ...
```

所有真实 LLM 调用都镜像到 `outputs/se_debug/<cmd>_<ts>/llm_calls.jsonl`，并在 stdout 给摘要，
便于事后核对 agent 的中间推理。

---

## ① feedback loop / log 落盘 → `handoff`

**要测什么**：interactive 的 feedback loop 有没有 bug、log 有没有正确落盘
（人 Finish 的成功信号、verbatim 人 feedback、final_code、chat_history）。

```bash
python -m capx.self_evolve.debug handoff <trial_dir>        # 或直接给 .json
python -m capx.self_evolve.debug handoff --logs-root logs   # 不给路径 = 取最新一条
python -m capx.self_evolve.debug handoff <trial_dir> --show-chat   # 额外 dump 整条 chat_history
```

- 走 `load_handoff()` 做**契约校验**：schema / task / final_code / chat_history 任一不合规 → 精确报错
  （这就是"log 落盘对不对"的判据）。
- 打印 success 信号落在哪个 attempt、settings、chat 轮数、**verbatim 人 feedback**、final_code。
- 不调用 LLM。

## ③ postprocessor → `postprocessor`

**要测什么**：能不能正确读入关键的 log；distill/rewrite agent 的处理结果符不符合预期。

```bash
# 默认 dry-run（写进临时 mem/，不碰真库），auto judge（离线无 env），跑 rewrite + distill
python -m capx.self_evolve.debug postprocessor <trial_dir> --model <m>

python -m capx.self_evolve.debug postprocessor <trial_dir> --skip-rewrite   # 只测 distill（隔离）
python -m capx.self_evolve.debug postprocessor <trial_dir> --judge interactive  # 逐个肉眼判 rewrite
python -m capx.self_evolve.debug postprocessor <trial_dir> --mem mem/        # 真写进库（去掉 dry-run）
```

- 经 `handoff.postprocessor_kwargs()` 把 handoff 喂给 `run_feedback_postprocessor`。
- 打印：输入摘要 → final_code → digest（带 `[#idx]`）→ 写出的 `history_pool` 成对文件 → llm_calls 路径。
- `--judge`：`auto`（全accept，让通用化跑满）/ `reject`（保留原码，隔离"不通用化"）/ `interactive`
  （终端逐个 y/n）。注意离线 judge **不执行 env**；真·env-in-the-loop rewrite 是 web 集成态的事。

## ④ skill management / planner → `planner`

**要测什么**：planner 能否**通过提供的工具看到 history pool**，并产出合理的 plan/判断。

```bash
python -m capx.self_evolve.debug planner --mem mem/ --tools-only   # 只看 History Reader 工具暴露了什么
python -m capx.self_evolve.debug planner --mem mem/ --force --model <m>   # 完整跑 proposer↔reviewer
```

- **先**打印 `list_unprocessed()` 与每条 `read_digest(id)`——即 planner 的工具**真正能看见**的 pool 视图；
  `--tools-only` 到此为止（不需要 LLM，先确认 pool 可见）。
- 再跑 `run_update_planner`（`--force` 可绕过 `trigger_history_count`），打印写出的 candidate 源码 +
  `stats.json` + reviewer feedback + llm_calls 路径。

---

## 注入式回调（可在脚本/测试里直接用）

`capx.self_evolve.debug` 暴露：

- `make_logged_query_fn(model, server_url=..., log_path=...)` → 真 LLM `query_fn`，记录每次 `{messages,
  content, reasoning}`（`fn.calls` 在内存里留档）。
- `auto_accept_judge()` / `reject_judge()` / `interactive_judge()` → `judge_fn` 替身（离线无 env）。

单测里则用**脚本化 fake `query_fn`** + `tmp_path` 的 `MemStore` 完全离线跑模块
（见 `tests/test_debug_harness.py`、`tests/test_feedback_postprocessor.py`、`tests/test_update_planner.py`）。

> ⑤ Benchmark Evaluator / ⑥ Heartbeat 的孤立驱动尚未接进本 CLI（它们更重，需 sim/cron）；
> 现有 `tests/test_benchmark_evaluator.py`、`tests/test_scheduler.py` 已覆盖其纯逻辑。
