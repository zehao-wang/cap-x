# Self-evolvable cap-x

本文档描述 cap-x 下一代 self-evolve agent 系统的设计。设计借鉴 openclaw
（参考 `/leonardo/home/userexternal/zwang003/Projects/openclaw`）的 memory /
heartbeat / cron 机制，目标是：在拿到 human feedback 之后，agent 有一套**可自我验证、
需人类批准、可终身更新**的流程来沉淀新能力。

---

## 1. 起点：cap-x agent0

我们的设计基于 **cap-x agent0**。它已具备：

- 基本的环境交互能力；
- 基本的 planning 能力；
- **VDM**（cap-x 内的一个 agent，用于自动评测 task progress / 判定成功失败）。

agent0 的能力分层：

- **primitives**：一组与环境交互的 API（最底层）。
- **skill library**：由 primitives 组合而成的通用函数。在 agent0 语境下**不涉及
  subtask**。

我们还实现了与人类的远端交互通信：可以选择数据集，默认 LLM 为服务器上的
`qwen3.6-27B`。

self-evolve 系统在 agent0 之上**新增 / 部分重新设计**，使其能把交互中得到的成功经验，
经过验证后固化为长期能力。

---

## 2. Concepts（术语表）

频繁用在后文设计里的基本概念。**术语必须全文一致**（见每条末尾的命名约定）。

1. **skill**（openclaw 语境）：一些可更新的、关注**特定 task 之间协作**的工作流。
   注意它**不是** skill library 里的内容，两者是不同概念。

2. **system prompt**：每次 LLM 调用都固定出现的内容，规定必须考虑的事情和固定流程。

3. **Memory**：分短期、长期。
   - **短期记忆 = `history_pool`**：人机交互中**成功**积累下来的经验（哪些能做 /
     不能做、code 层面哪些与人协作的更改让任务成功了）。**只有成功的 history 进入短期
     记忆**，且**只被 Library Management 消费**。
   - **长期记忆**：经过完整 benchmark evaluation 验证、确实对 task success 有贡献的
     coding 层产物（code / config / hyper-param），落在 `primitives`、`skill_library`、
     `atomic_task_library` 中。

4. **Heartbeat**：定时唤起一些 daily task 或长期 plan 的 task，是推动模型演进的机制
   （参考 openclaw 实现）。长期目标与 daily task 见 §6。

5. **Cron job**：在规定时间触发 heartbeat，用于固定时刻 / 需固定等待的任务。例如夜间
   固定时间触发 Benchmark Evaluator 进行长期记忆更新（参考 openclaw 实现）。

6. **Memory Compact**：未来设计，目前不考虑。

7. **Human feedback**：interactive 窗口接收人类反馈。当前仅 text；未来需支持 rgbd
   video。届时会有一个专门的 skill 处理 raw rgbd demo 视频：video caption（简短但保留
   subtask 顺序）、hand tracking（mediapipe）等。

8. **Atomic task library**（新概念）：对长期出现在各任务里的**原子任务**，其流程可复用。
   例如 `pick` 一个物体需要考虑的内容是相似的：先 approach → double-check 是否已在正确
   pre-grasp 位姿 → 夹取 → 判断物体是否在夹爪内……
   - **粒度规则（重要，LLM-2 审核的核心标准）**：atomic task **不与具体物体 / 具体任务
     绑定**。允许 `pick(obj)` / `place(...)` 这种通用原子任务；**拒绝** `put_apple()`
     这种绑定具体物体或具体任务的函数。
   - **数据模型**：一个 atomic task = **函数代码 + 一份 config（超参）**。例如 grasp
     相关的 margin 作为 grasp atomic task 的 config 字段（见 §4 Feedback Postprocessor）。
   - **命名约定**：全文统一用 **atomic task library**。（旧文档中出现过的 “subtask
     library” 即指此物，已废弃该叫法。）

9. **Feedback Postprocessor**：当一次任务**依赖 human feedback 给的额外信息**才成功后启动
   的流程，负责把代码**通用化 / 超参化**，去除对这次一次性额外信息的直接依赖。详见 §4。

10. **VDM**：见第 1 节——cap-x 内用于自动评测的 agent。Benchmark Evaluator 中继续用它做
    自动成功判定。

---

## 3. 端到端 Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ [Live Loop]  human-in-the-loop 执行任务 (主交互循环)                    │
│   人判定 success  ← 唯一的 success signal，必须人给                      │
└───────────────┬─────────────────────────────────────────────────────┘
                │ 若本次依赖了 human feedback 的额外信息才成功
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [Feedback Postprocessor]                                              │
│   多轮 rewrite（仍与环境交互），把代码通用化 / 超参化                     │
│   每轮只由人判对错（不再接收细节 feedback）：对→结束，错→重写             │
└───────────────┬─────────────────────────────────────────────────────┘
                │ final_code
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ mem/history_pool/   短期记忆，仅成功 history，暂不删除（openclaw 式追加）│
└───────────────┬─────────────────────────────────────────────────────┘
                │ 触发：未处理 history 数 ≥ trigger_history_count（默认 5）
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [Update Planner]  (Library Management 的一部分)                        │
│   LLM-1: 读未处理 history+task → 提议新增/更新代码                      │
│   LLM-2: 审核(去重 / bug / general / 粒度 ≤ atomic task)  ──revise──┐  │
│          revise 上限 = max_revise_iterations；超限则强制定稿         │  │
│          └──pass──► 由 LLM-2 写入 func_candidate_pool ◄──────────────┘  │
│   完成后：标记这批 history 为已处理（history 本身保留）                  │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ mem/func_candidate_pool/   候选函数 + 累积统计（统计跟随 candidate）     │
└───────────────┬─────────────────────────────────────────────────────┘
                │ 触发：夜间 cron 固定一次（也可人工唤醒）；pool 非空时执行
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [Benchmark Evaluator]  (短期→长期记忆的机制；用 VDM 自动评测)            │
│   把 candidate 注入“可选工具”，在 sim 里跑 benchmark                    │
│   累积统计（每个 candidate，跨 eval 持续累加；不记 negative）：           │
│     · 终版成功代码里用到      → positive += 1                          │
│     · 本轮被调用过(chat 或终版) → used_runs += 1                        │
│     · 每经历一轮 eval         → eval_runs += 1                          │
│   生成报告 → 通知用户：建议固化哪些 / 建议遗弃哪些                       │
│   ★ 必须用户批准；未经允许不得 update                                   │
└───────────────┬─────────────────────────────────────────────────────┘
                │ 用户批准后
                ▼
   固化(promote): positive ≥ promote_threshold(默认 10) 且 eval_runs ≥ min_samples
                  → 写入 capx/skill_library/ 或 capx/atomic_task_library/
                    （函数 docstring 记更新历史日期，不写具体改了什么）
   遗弃(delete) : eval_runs > max_idle_evals 且 used_runs == 0
                  且无其他存活 candidate 引用它 → 从 pool 移除
   保留(keep)   : 其余情况（被用过 / 未达阈值 / 仍被引用）→ 留在 pool
```

---

## 4. 各模块设计

### 4.1 Live Loop & Success Signal

主交互循环以 human-in-the-loop 方式执行任务，直到成功。

- **success signal 在新设计里只允许人给**：VDM 不在 live loop 里判定成功；成功与否由
  用户在 interactive 窗口确认。
- 仅当用户确认成功，才进入 Feedback Postprocessor → 写入 `history_pool`。

### 4.2 Feedback Postprocessor

由于 human feedback 常带额外信息，我们在尝试时应**先尽量用这些信息让任务成功**。成功之后，
Feedback Postprocessor 负责把代码改得更通用：

- **流水线位置**：在「人判定成功」与「生成 final_code 写入 history_pool」**之间**。即进入
  短期记忆的已经是通用化后的版本。
- **做法**：多轮 code rewrite，去除对这次一次性额外信息的直接依赖。
  - 例：feedback 要求 eef 始终与 `z=0` 平面保持 `0.03` 距离。第一版可能直接在 main 里
    `z += 0.03`；rewrite 的更通用形态是——把这个 margin 变成 grasp 相关的超参，成为 grasp
    atomic task config 的一部分，通用于所有情况。
- **验证方式**：rewrite **仍需与环境交互**实际执行；**每轮只由人判对错**（不再接收细节
  feedback）。对 → 结束；错 → 继续重写。

### 4.3 Library Management — Update Planner

Library Management 管理并更新 `atomic_task_library` 与 `skill_library`。Update Planner 是其
前半段。

- **触发**：当 `history_pool` 中**未处理**的 history 数达到 `trigger_history_count`（默认 5）
  自动开始（比 Benchmark Evaluator 更频繁）。
- **不能无限全读**：Library Management 必须记录已处理过哪些 history（见 §5 的
  `.processed_history`），每次只处理增量。history 本身保留、暂不删除。
- **流程**：
  1. **LLM-1**：input = 这批未处理 history 及其对应 task；output = 一些应新增 / 更新的代码。
  2. **LLM-2**：基于 LLM-1 的产出，结合**当前 library** 审核：是否重复功能、是否引入 bug、
     是否够 general、粒度是否超出 atomic task（拒绝 `put_apple()` 这类绑定具体物体/任务的
     函数）。
     - pass → **由 LLM-2 把函数写入 `func_candidate_pool`**。
     - 不 pass → 带着 feedback 唤醒 LLM-1 去 revise。
  3. **终止**：revise 轮数上限 = `max_revise_iterations`（config）。几轮 feedback + 查错后
     若仍未收敛，**强制要求模型定稿一个方案写入 `func_candidate_pool`**。
  4. 处理完这批后，标记其为已处理；这批 history 的对话已无进一步意义（但文件保留）。

### 4.4 Library Management — Benchmark Evaluator

Update Planner 的产物是 `func_candidate_pool` 的更新；Benchmark Evaluator 是**短期记忆 →
长期记忆**的固化机制。

- **触发**：夜间 cron 固定一次；也可**人工唤醒**。仅当 `func_candidate_pool` 非空时执行。
- **评测**：类似 `scripts/run_agent0_qwen36_robosuite.sh` /
  `scripts/run_agent0_qwen36_libero.sh`，在模拟器里跑 benchmark 数据集，用 **VDM** 做自动
  成功判定。区别在于：`func_candidate_pool` 的内容会作为**可选工具**加入（注入方式见 §7），
  而不仅是已固化的 primitives / skill_library / atomic_task_library。
- **统计（每个 candidate，跨多次 eval 持续累积，跟随 candidate 直到其被删除；不记 negative）**：
  - `positive`：在最终成功代码里出现的累计 task 数（终版用到即 +1）。
  - `used_runs`：本轮 eval 中被调用过（chat 或终版任意处）则 +1。
  - `eval_runs`：candidate 进入 pool 后经历的 Benchmark Evaluator 次数，每轮 +1。
- **决策**：
  - **固化（promote）**：`positive ≥ promote_threshold`（默认 10，即在 ~10 个 task 的最终成功
    代码里都出现）且 `eval_runs ≥ min_samples`（默认 5）→ 写入对应长期 library。被固化函数的
    **docstring 要记更新历史日期**（不写具体改了什么）。
  - **遗弃（delete）**：candidate 在 pool 中经历 `eval_runs > max_idle_evals` 次评测**始终
    没被调用**（`used_runs == 0`）**且没有其他存活 candidate 引用它** → 从
    `func_candidate_pool` 删除。引用关系在删除前**动态扫描** pool 内其余 candidate 源码判定。
  - **保留（keep）**：其余情况——被用过但未达固化阈值、或仍被其他存活 candidate 引用 →
    暂留在 `func_candidate_pool`。
- **用户批准（强约束）**：Benchmark Evaluator 完成后生成一份**简单报告**，通知用户它认为
  哪些该固化、哪些该遗弃。**任何对长期 library 的 update / delete 都必须经用户批准；未经
  允许不得执行。**

> 设计取舍：我们**不记录 negative**（函数被试后弃用未必代表它差，可能只是该 task 不适用）。
> 固化只看 `positive` 阈值；遗弃只针对「长期占着 pool 又从没被用、且无人依赖」的 candidate；
> 最终都由用户把关。

---

## 5. 存储布局与数据 Schema（`mem/`）

所有 pool 都与 memory 相关，统一放在仓库的 `mem/` 目录下。

```
mem/
├── history_pool/                # 短期记忆：成功交互 log，openclaw 式追加，暂不删除
│   └── <task>__<YYYYMMDD-HHMMSS>.json
├── func_candidate_pool/         # 候选函数（由 LLM-2 写入）+ 累积统计
│   ├── <func_name>.py           #   候选函数源码
│   └── <func_name>.stats.json   #   该函数的累积统计
└── .processed_history           # Library Management 已处理的 history id 列表(增量游标)
```

长期 library（固化目标，作为可 import 的代码模块，放在 `capx/` 包内；保持模块化但不过度
模块化）：

```
capx/skill_library/*.py          # 由 primitives 组成的通用函数
capx/atomic_task_library/*.py    # 每个文件 = atomic task 的函数 + 其 config(超参)
```

> 注意：候选**不直接改** `capx/skills/library.py`；本系统沉淀的新能力写入上面这两个**新模块**。

### success log schema（`history_pool/*.json`）

```jsonc
{
  "task": "<task 名字>",
  "settings": { /* 用的设置：env / dataset / llm 等 */ },
  "final_code": "<Feedback Postprocessor 通用化后的最终代码>",
  "chat_history": [ /* 完整对话，含 human feedback */ ],
  "datetime": "YYYY-MM-DD HH:MM:SS"
}
```

### candidate stats schema（`func_candidate_pool/*.stats.json`）

```jsonc
{
  "func_name": "<name>",
  "target_library": "skill_library | atomic_task_library",
  "positive": 0,                 // 在最终成功代码里出现的累计 task 数（≥promote_threshold 建议固化）
  "eval_runs": 0,                // 进入 pool 后经历的 Benchmark Evaluator 次数
  "used_runs": 0,                // 其中至少被调用过一次的 run 数（==0 且 eval_runs>max_idle_evals 触发遗弃判定）
  "source_history": ["<id>"],    // 来源 history
  "created_date": "YYYY-MM-DD",
  "last_eval_date": "YYYY-MM-DD"
}
```

> 引用关系（“是否有其他存活 candidate 依赖本函数”）在遗弃判定时**动态扫描** pool 内其余
> candidate 源码得出，不冗余存储。

---

## 6. Heartbeat / Cron：长期目标与 Daily Task

- **长期目标**：① 提升在 benchmark 上的效果；② 在人机交互的 task 上更快、更容易成功。
- **Daily task（由 heartbeat 唤起）**：把之前**正确率低**的 task 拎出来交给人 feedback，
  通过新产生的成功 history 推动 library 更新。
- **Cron**：在固定时刻触发 heartbeat。典型用法——夜间固定时间触发 Benchmark Evaluator 做
  长期记忆更新（同时 Update Planner 由 history 增量阈值自行更频繁触发）。

---

## 7. 与现有代码的对接

- **可选工具注入点**：candidate 复用现有 `capx/skills/library.py`（`SkillLibrary`）暴露的
  两个机制——
  - `get_skill_docs()`：把候选函数文档拼进与已固化 skill **同一 prompt 区域**；
  - `inject_into_namespace()`：把候选函数注入可执行命名空间。
  Benchmark Evaluator 在 sim 评测时，用这两个机制把 `func_candidate_pool` 一并提供给 agent。
- **固化目标 ≠ `library.py`**：现有 `library.py` 的「按出现频次 promote」逻辑可继续保留，但
  self-evolve 沉淀的新能力固化到 `capx/skill_library/` 与 `capx/atomic_task_library/`
  两个新模块，以保持模块化。

---

## 8. 配置（hyper-params）

集中放在 config 里，便于调参：

| 配置项 | 默认 | 含义 |
| --- | --- | --- |
| `update_planner.trigger_history_count` | 5 | `history_pool` 未处理数达到即自动触发 Update Planner |
| `update_planner.max_revise_iterations` | 5 | LLM-1↔LLM-2 revise 轮数上限，超限强制定稿写入 candidate pool |
| `benchmark_eval.min_samples` | 5 | 建议固化前所需的最小 `eval_runs` |
| `benchmark_eval.promote_threshold` | 10 | `positive` 达到即建议固化（在 ~10 个 task 最终成功代码里都出现） |
| `benchmark_eval.max_idle_evals` | 50 | candidate 经历这么多次 eval 仍 `used_runs==0` 且无人引用 → 建议遗弃（设大些，因为一整个 task set 可能都用不到某 candidate） |
| `benchmark_eval.schedule` | 夜间 cron | 固定一次；也可人工唤醒 |

---

## 9. Human interaction logic（interactive 交互循环设计）

interactive（human-in-the-loop）主交互循环的设计如下。核心原则：**一次只围绕同一个
trial 反复修改，success 只由人给，每次 feedback 默认重置场景重来**。实现在
`capx/web/async_trial_runner.py`（循环）+ `capx/envs/base.py`（状态快照/恢复）。

### 9.1 每轮流程

```
模型生成代码 → 执行 → 展示执行结果 → 暂停等人（永远在“执行之后”）
        ↑                                              │
        └──────── feedback：reset 场景 + 重生成 ◄───────┘
```

- **暂停点永远在一次代码执行之后**：人看到执行结果后才被要求给反馈（不会在没有执行结果时
  弹 feedback）。
- **无限等待，无超时 / 无定时重启**：暂停时一直等人，不存在“N 秒后自动重来”。Stop 按钮通过
  取消任务来中断。

### 9.2 人的动作（只有两个）

- **Finish**：人确认成功并结束 trial。**这是 interactive 模式唯一的 success signal**
  （§4.1）；模型自己的 “FINISH” 在 interactive 模式下**永不结束** trial。
- **发 feedback（有文字）**：触发“reset + 重来”（见 §9.3）。
- 空提交 / 无文字的 send = **no-op**（继续等待）。feedback **不长期保留**：只有当轮的
  feedback 进入下一次生成的 context，不累积历史 feedback。

### 9.3 feedback 默认 reset 重来

每次 feedback 都**默认把场景 reset 回这个 episode 的初始状态**，然后用
`[task prompt + 上一轮的代码 + 本次 feedback]` 作为 context **重新生成一份完整的尝试**并执行。

- **没有增量 multi-turn**：interactive 模式不再做“执行一块→模型决定 regenerate/finish→再执行
  下一块”的增量推进；每次 feedback 就是一次干净的全量重试。
- **没有单独的 Reset 按钮**：reset 已经是 feedback 的默认行为。
- context 里**只额外加“上一轮代码 + feedback”**，不带完整对话历史、也不再注入多步脚手架
  prompt（feedback 以独立 user 消息进入对话）。
- 模型若没产出代码：给出提示并回到暂停等下一次 feedback（不退出）。

### 9.4 reset 机制：MuJoCo 全量状态快照 / 恢复

- **仅用于模拟器数据集**：reset 把模拟器强制送回**这个 episode 刚开始的状态**（不重采样新
  episode）。
- 实现：episode 开始（初始 reset 之后）用 `mujoco.mj_getState(..., mjSTATE_INTEGRATION)`
  抓**全量积分状态**（time/qpos/qvel/act/ctrl/mocap/applied-forces/warmstart）+ 少量
  Python 侧记账；reset 时 `mj_setState` + `mj_forward` 精确恢复。对 robosuite 与 LIBERO
  通用（都走 robosuite 的 `MjSim`）。
- 恢复时**同步清空 viser 的 `frame_history`**（与正常 `reset()` 一致），保证回放时间轴每次
  尝试都是干净、独立的（否则旧帧会污染滑块、相机视图错乱）。

### 9.5 prompt 拼装

- feedback / reset note 作为**独立的 user 消息**进入对话（不拼进多步模板字符串）。
- **发送前折叠连续同角色消息**：上述拼装可能产生相邻的多条 user 消息，发送给 LLM 前合并为
  一条，避免要求严格角色交替的 chat-template 服务端报错。

### 9.6 与 headless / benchmark 的关系

- **headless / 自动 benchmark 评测保持原样**：仍由模型自己驱动 `REGENERATE / FINISH`，无人
  在环、无 reset。上述改动只作用在 interactive（`await_user_input_each_turn` 为真）。


