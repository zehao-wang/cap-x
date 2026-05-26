# Live Loop & Interactive 交互循环

> **状态**: ✅ 已实现
> **读 → 写**: human-in-the-loop 交互 → 人确认成功的 trial（其对话/代码喂给
> [03-feedback-postprocessor.md](03-feedback-postprocessor.md)）
> **实现**: `capx/web/async_trial_runner.py`（循环）+ `capx/envs/base.py`（状态快照/恢复）
> **依赖契约**: [concepts.md](concepts.md)；可视化部分见 [02-visualization.md](02-visualization.md)

## 1. Live Loop & Success Signal

主交互循环以 human-in-the-loop 方式执行任务，直到人确认成功。

- **success signal 只允许人给**：模型自己的 `FINISH` **永不结束** trial；成功与否由用户在
  interactive 窗口点 **Finish** 确认。
- 仅当用户确认成功，才进入 [Feedback Postprocessor](03-feedback-postprocessor.md) →
  写入 `history_pool`。

核心原则：**模型自己跑完整的多轮自纠错（multi-turn），结束后人来评判；人 feedback 默认
reset 场景、基于反馈重跑整轮**。

## 2. 两层循环

interactive 在 **模型自驱 multi-turn** 外面套一层 **人类循环**：

```
   ┌──────────────── 一个 model multi-turn pass ────────────────┐
   │  生成代码 → 执行一块 → (视觉反馈/差分) → 模型决策            │
   │      ↑                                          │           │
   │      └──── REGENERATE：替换剩余代码（增量，保留状态） ◄────┘ │
   └───────────────────────────┬───────────────────────────────┘
                               │ pass 结束（FINISH / 到 limit / 报错·超时 / episode terminated）
                               ▼
                       暂停，等人评判
                   ┌───────────┴───────────┐
              Finish=成功收尾        feedback：reset 场景 + 基于反馈重跑整轮
                                          （空发送=继续等）
```

- **内层（模型自驱 multi-turn）**：执行一个 code block 后，模型在 `REGENERATE / FINISH`
  脚手架下决定是改写剩余代码（增量推进，**保留已执行状态**）还是结束。受 `MULTITURN_LIMIT`
  约束。决策前的环境判定有两条规则：
  - **报错（`sandbox_rc != 0`，stderr 有 traceback）直接走修代码、不经过 VDM**：跳过视觉差分，
    并在决策 prompt 里明确要求"修复错误、不要 FINISH"；即使模型仍回 FINISH 也强制转成
    regenerate——**硬错误永远不能被判成 finish/success**。
  - **VDM 视觉差分会附上 console stdout 做 grounding**：单一固定相机视角常看不出"是否真的抬起"
    这类细微变化，所以把代码打印的 stdout（如 `Lifted the cube`、测得位姿）作为"agent 自报、需与
    图像互证"的上下文一并喂给 VDM，减少误判。
- **外层（人类循环）**：仅 interactive 有。模型那一轮 multi-turn 结束后才暂停问人。

## 3. 何时暂停问人 + 人的动作

**暂停点 = 模型那一轮 multi-turn 结束之时**，只有两种到达方式（见 §2）：

1. 模型自己 `FINISH`（它认为成功了）；
2. 模型因各种原因结束但未成功：到达 `MULTITURN_LIMIT`、报错/超时、或 episode terminated。

无论哪种，都暂停并要求人评判。人的动作只有：

- **Finish**：人确认成功并结束 trial。**这是 interactive 唯一的 success signal**（§1）。
- **发 feedback（有文字）**：触发"reset + 基于反馈重跑整轮"（见 §4）。
- **空提交 / 无文字的 send = no-op**（继续等）。**无限等待，无超时 / 无定时重启**；Stop 按钮
  通过取消任务来中断。

## 4. feedback 默认 reset 重来

每次 feedback 都**默认把场景 reset 回这个 episode 的初始状态**，然后用
`[任务 prompt（含 API/工具说明）+ 上一轮全部代码 + 本次 feedback]` 作为 context **重新跑一整轮
模型 multi-turn**（模型可以在新一轮里再次 REGENERATE 多次）。

- **没有单独的 Reset 按钮**：reset 已经是 feedback 的默认行为。
- feedback **不长期保留**：只有当轮 feedback 进入下一轮的 context，不累积历史 feedback。
- 模型若没产出代码：给出提示并回到暂停等下一次 feedback（不退出）。

## 5. reset 机制：MuJoCo 全量状态快照 / 恢复

- **仅用于模拟器数据集**：reset 把模拟器强制送回**这个 episode 刚开始的状态**（不重采样新
  episode）。
- 实现：episode 开始（初始 reset 之后）用 `mujoco.mj_getState(..., mjSTATE_INTEGRATION)`
  抓**全量积分状态**（time/qpos/qvel/act/ctrl/mocap/applied-forces/warmstart）+ 少量
  Python 侧记账；reset 时 `mj_setState` + `mj_forward` 精确恢复。对 robosuite 与 LIBERO
  通用（都走 robosuite 的 `MjSim`）。
- 恢复时**为 viser `frame_history` 开启一段新 segment（`new_segment()`，不再 `clear()`）**：
  每次重跑 = 一段独立 trail，先前的仍可回看。一次全新 trial 用正常 `reset()` →
  `frame_history.clear()` 丢弃所有 segment。回放细节见 [02-visualization.md](02-visualization.md)。

## 6. prompt 拼装

- **模型 multi-turn 用「多步脚手架」(`multi_turn_prompt`，REGENERATE/FINISH 模板)**：interactive
  与 headless **一致使用**——这是模型自驱多轮的核心。决策 prompt 由 `clean_base_prompt`（干净
  任务 prompt）+ 已执行代码/console 输出 + 可选视觉反馈/差分 拼成。
- **feedback 重跑的 context 区别对待**：用 `clean_base_prompt`（**保留 API/工具说明等关键信息**，
  否则模型不知道能调用哪些 tool）+ 上一轮全部代码（reset note 里）+ feedback。三者作为独立
  user 消息追加，**只省略上一轮的思考/中间过程**，不带完整对话历史。
- **发送前折叠连续同角色消息**：上述拼装可能产生相邻的多条 user 消息，发送给 LLM 前合并为
  一条，避免要求严格角色交替的 chat-template 服务端报错（`_merge_consecutive_messages`）。

## 7. 与 headless / benchmark 的关系

- **headless / 自动 benchmark 与 interactive 共用同一套模型自驱 multi-turn**（`while True` 内层
  循环）。区别只有外层：headless 在模型 `FINISH`/到 cap 时**直接结束** trial（success = 环境
  reward 或 `num_finishes>0`）；interactive 在同一点**暂停问人**，success = `human_finished`。

## 8. 交互行为日志（debug / 追溯）

每个 trial 把 agent↔LLM 的每次交互与 tool 调用流结构化落盘到 `output_dir/trial_XX/`，便于查询和
追溯模型行为（实现 `capx/utils/trace_logger.py`，接进 `async_trial_runner.py`）：

- **`llm_trace.jsonl`**：每次 LLM 调用一条记录——`phase`（initial / multi_turn / feedback_retry /
  env_description / img_differencing）、`turn`、`model`、耗时、**完整输入 messages** 与
  **输出 content + reasoning**（REGENERATE/FINISH 决策与生成的代码都在 output content 里）。输入里
  的内联图片会被抽出存成 `trace_images/llmNNN_imgK.png`，JSONL 里只留引用，保持文件小、易 `jq` 查询。
- **`events.jsonl`**：把 LLM 调用与 tool 执行步骤（`execution_logger` 的 step）**按时间顺序合并**，
  实时追加，清晰看到「tool 调用流 + 每轮对话」的真实交织顺序。
- **`trace.md`**：trial 结束（含成功 / Stop 取消 / 报错任一退出路径，写在 `finally` 里）时，
  由 `events.jsonl` 渲染出的人类可读时间线。
