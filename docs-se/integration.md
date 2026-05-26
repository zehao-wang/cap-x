# 与现有代码的对接

> **共享契约** — 主要被 [05-benchmark-evaluator.md](05-benchmark-evaluator.md) 使用（评测时把
> 候选函数提供给 agent）。

- **可选工具注入点**：candidate 复用现有 `capx/skills/library.py`（`SkillLibrary`）暴露的
  两个机制——
  - `get_skill_docs()`：把候选函数文档拼进与已固化 skill **同一 prompt 区域**；
  - `inject_into_namespace()`：把候选函数注入可执行命名空间。
  Benchmark Evaluator 在 sim 评测时，用这两个机制把 `func_candidate_pool` 一并提供给 agent。
- **固化目标 ≠ `library.py`**：现有 `library.py` 的「按出现频次 promote」逻辑可继续保留，但
  self-evolve 沉淀的新能力固化到 `capx/skill_library/` 与 `capx/atomic_task_library/`
  两个新模块，以保持模块化（路径见 [storage.md](storage.md)）。
