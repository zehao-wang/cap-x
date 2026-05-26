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

---

## 核心原则：digest 优先，按需 drill-down

`history_pool/*.json` 每条都带完整 `chat_history`（含 human feedback），单条就可能很长；未处理
批次 ≥ `trigger_history_count` 时全文拼进 context 会撑爆窗口。因此 **LLM-1 和 LLM-2 都默认只读
digest**，仅在需要核实时才钻回全文：

- **入口 = digest**：读 [03-feedback-postprocessor.md](03-feedback-postprocessor.md) 在入 pool
  前产出的 `<id>.digest.md`（每条几百字、带 `[#idx]` 回溯标注，schema 见 [storage.md](storage.md)），
  **不是** `chat_history` 全文。digest 已替模型点出"值得沉淀什么"。
- **按需 drill**：仅当某论断需核实（pattern 是否真成立 / 引用是否属实 / 粒度是否够 general）时，
  才用下面的只读工具按 `[#idx]` 取 `.json` 里那几轮原文，每次只取相关片段、不整段灌入。
- **硬上限** `update_planner.max_read_iterations`（默认 20）：逼近上限时必须基于已读证据定稿，
  不再探索。这是 cap-x 自有实现，**不**依赖外部 `adb` CLI。

### History Reader 工具集（只读）

| 工具 | 作用 |
| --- | --- |
| `list_unprocessed()` | 返回未处理 history 的轻量索引：`{id, task, datetime, digest_path}`，**不返回正文**。模型的入口。 |
| `read_digest(id)` | 读某条 history 的 `<id>.digest.md`（小、限长、带 `[#idx]`）。**默认就读这个**。 |
| `read_history(id, field, offset?, limit?)` | **drill-down**：读 `.json` 全文的某字段（`final_code` / `chat_history` / `settings`）；对长字段用 `offset`/`limit` 分页，按 digest 的 `[#idx]` 精准取那几轮。 |
| `grep_history(regex, ids?, field?)` | 跨 history 正则检索（工具名、API 名、错误关键字、引用的物体名…），返回命中位置 `{id, field, message_index}`，供精准定位。 |
| `read_library(target?)` | 读**当前** `skill_library` / `atomic_task_library` 的函数签名+docstring（用于查重、判 general、对粒度），同样按需、不全量灌入。 |
| `propose(...)` | 终结工具（类比 `complete_task`）：提交结构化提案后结束本轮循环（见下「结构化产出」）。 |

### 单次调用内的工作流（phases；iter 区间是指引不是硬门）

1. **Skim digests**（≈ iter 1-5）：`list_unprocessed()` 拿索引 → 对每条 `read_digest(id)`。
2. **Drill on demand**（≈ iter 6-15，**仅在需要时**）：某条 REUSABLE PATTERN 要落成 candidate
   但证据不足 / 粒度存疑时，按 `[#idx]` 用 `read_history(...)` 钻回原文核实，跨 history 找共性用
   `grep_history`。digest 已足够就跳过本阶段。
3. **Finalize**（iter ≤ max）：调一次 `propose(...)` 收尾。

---

## 流程

两个 LLM 都跑在上面的 History Reader 模式（digest 优先、按需 drill）：

1. **LLM-1（提议）**：`propose` 出一批应**新增 / 更新**的候选函数草案（每条带 `source_history`
   引用 + rationale，见下）。
2. **LLM-2（审核）**：审核 LLM-1 的产出，drill 回原始 history 与当前 library 核对——
   `source_history` 引用是否属实、是否重复功能、是否引入 bug、是否够 general、粒度是否超出
   atomic task（拒绝 `put_apple()` 这类绑定具体物体/任务的函数——粒度规则见 [concepts.md](concepts.md)）。
   - pass → **由 LLM-2 把函数写入 `func_candidate_pool`**（同时创建 `*.stats.json`，schema 见
     [storage.md](storage.md)；`source_history` 直接取自提案的引用）。
   - 不 pass → 带着 feedback 唤醒 LLM-1 去 revise（LLM-1 可再用工具补读它之前没看的片段）。
3. **终止**：revise 轮数上限 = `update_planner.max_revise_iterations`（默认 5）。几轮 feedback
   + 查错后若仍未收敛，**强制要求模型定稿一个方案写入 `func_candidate_pool`**。
   注意这与 step-1/2 内部的 `max_read_iterations`（单次调用内的读取预算）是两个独立上限。
4. 处理完这批后，标记其为已处理；这批 history 的对话已无进一步意义（但文件保留）。

### 结构化产出（`propose` payload）

LLM-1 的 `propose` / LLM-2 定稿都用同一结构（类比 Agent Debugger 的 `complete_task` JSON 契约，
便于自动落盘、可校验）：

```jsonc
{
  "candidates": [
    {
      "func_name": "<name>",
      "target_library": "skill_library | atomic_task_library",
      "code": "<函数源码草案>",
      "source_history": ["<history id>", ...],   // 证据来源，落入 stats.json
      "rationale": "<为什么 general / 为什么这个粒度>"
    }
  ]
}
```

- `target_library` / `source_history` 直接对应 `*.stats.json` 的字段（见 [storage.md](storage.md)）。
- `source_history` 必须引用 `list_unprocessed()` 给出的真实 id；LLM-2 会回查核对，杜绝编造。
- `candidates` 可为空（这批 history 没有值得沉淀的通用能力）。
