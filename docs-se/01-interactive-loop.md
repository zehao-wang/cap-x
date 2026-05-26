# Live Loop & Interactive 交互循环

> **状态**: ✅ 已实现
> **读 → 写**: human-in-the-loop 交互 → 人确认成功的 trial（其对话/代码喂给
> [03-feedback-postprocessor.md](03-feedback-postprocessor.md)）
> **实现**: `capx/web/async_trial_runner.py`（循环）+ `capx/envs/base.py`（状态快照/恢复）
> **依赖契约**: [concepts.md](concepts.md)；可视化部分见 [02-visualization.md](02-visualization.md)

## 1. Live Loop & Success Signal

主交互循环以 human-in-the-loop 方式执行任务，直到成功。

- **success signal 在新设计里只允许人给**：VDM 不在 live loop 里判定成功；成功与否由
  用户在 interactive 窗口确认。
- 仅当用户确认成功，才进入 [Feedback Postprocessor](03-feedback-postprocessor.md) →
  写入 `history_pool`。

核心原则：**一次只围绕同一个 trial 反复修改，success 只由人给，每次 feedback 默认重置
场景重来**。

## 2. 每轮流程

```
模型生成代码 → 执行 → 展示执行结果 → 暂停等人（永远在"执行之后"）
        ↑                                              │
        └──────── feedback：reset 场景 + 重生成 ◄───────┘
```

- **暂停点永远在一次代码执行之后**：人看到执行结果后才被要求给反馈（不会在没有执行结果时
  弹 feedback）。一次模型生成（attempt）可能含多个 code block；interactive 模式会**先把这次
  attempt 的所有 block 跑完再暂停**（不在 block 之间停）。
- **无限等待，无超时 / 无定时重启**：暂停时一直等人，不存在"N 秒后自动重来"。Stop 按钮通过
  取消任务来中断。

## 3. 人的动作（只有两个）

- **Finish**：人确认成功并结束 trial。**这是 interactive 模式唯一的 success signal**（见 §1）；
  模型自己的 "FINISH" 在 interactive 模式下**永不结束** trial。
- **发 feedback（有文字）**：触发"reset + 重来"（见 §4）。
- 空提交 / 无文字的 send = **no-op**（继续等待）。feedback **不长期保留**：只有当轮的
  feedback 进入下一次生成的 context，不累积历史 feedback。

## 4. feedback 默认 reset 重来

每次 feedback 都**默认把场景 reset 回这个 episode 的初始状态**，然后用
`[task prompt + 上一轮的代码 + 本次 feedback]` 作为 context **重新生成一份完整的尝试**并执行。

- **没有增量 multi-turn**：interactive 模式不再做"执行一块→模型决定 regenerate/finish→再执行
  下一块"的增量推进；每次 feedback 就是一次干净的全量重试。
- **没有单独的 Reset 按钮**：reset 已经是 feedback 的默认行为。
- context 里**只额外加"上一轮代码 + feedback"**，不带完整对话历史、也不再注入多步脚手架
  prompt（feedback 以独立 user 消息进入对话）。
- 模型若没产出代码：给出提示并回到暂停等下一次 feedback（不退出）。

## 5. reset 机制：MuJoCo 全量状态快照 / 恢复

- **仅用于模拟器数据集**：reset 把模拟器强制送回**这个 episode 刚开始的状态**（不重采样新
  episode）。
- 实现：episode 开始（初始 reset 之后）用 `mujoco.mj_getState(..., mjSTATE_INTEGRATION)`
  抓**全量积分状态**（time/qpos/qvel/act/ctrl/mocap/applied-forces/warmstart）+ 少量
  Python 侧记账；reset 时 `mj_setState` + `mj_forward` 精确恢复。对 robosuite 与 LIBERO
  通用（都走 robosuite 的 `MjSim`）。
- 恢复时**为 viser `frame_history` 开启一段新 segment（`new_segment()`，不再 `clear()`）**：
  每个 attempt = 一段独立 trail，先前 attempt 仍可回看。一次全新 trial 用正常 `reset()` →
  `frame_history.clear()` 丢弃所有 segment，重新开始。回放细节见
  [02-visualization.md](02-visualization.md)。

## 6. prompt 拼装

- feedback / reset note 作为**独立的 user 消息**进入对话（不拼进多步模板字符串）。
- **「多步模板」(`multi_turn_prompt`，REGENERATE/FINISH 脚手架) 只用于 headless/benchmark**
  （见 §7 的模型自驱多轮）。interactive 不注入它，也**不再依赖它存在**才暂停等人。
- **发送前折叠连续同角色消息**：上述拼装可能产生相邻的多条 user 消息，发送给 LLM 前合并为
  一条，避免要求严格角色交替的 chat-template 服务端报错。

## 7. 与 headless / benchmark 的关系

- **headless / 自动 benchmark 评测保持原样**：仍由模型自己驱动 `REGENERATE / FINISH`，无人
  在环、无 reset。上述改动只作用在 interactive（`await_user_input_each_turn` 为真）。
