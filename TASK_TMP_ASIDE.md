# 长期目标 / Long-term Goals

记录"暂时不做、但将来可能需要做"的事项。每条注明背景、现状、触发条件、范围，做完后移除或标记。

---

## 1. 把 web runner 的 VDM / 错误处理改进同步到 headless benchmark

- **背景**：在线 web runner（`capx/web/async_trial_runner.py`）已对模型自驱 multi-turn 决策做了两项改进：
  1. VDM 视觉差分 prompt 附带 console stdout 做 grounding（`capx/web/trial_support.py:build_state_diff_prompt`）——单视角看不清"是否真的抬起"时，用代码自报的 stdout 互证。
  2. 硬错误（`sandbox_rc != 0`，stderr 有 traceback）**跳过 VDM**、直接让模型按 traceback 修代码；且硬错误**绝不判 finish/success**（即使模型回 FINISH 也强制转 regenerate）。
- **现状**：headless / 自动 benchmark 走的是另一套 `capx/envs/trial.py`（`_handle_multi_turn_step` 等），**暂未同步，保持原样**。
- **触发条件**：web 端这两项改进在实跑中验证有效后，再镜像到 `trial.py`，让 benchmark 评测口径与在线一致。
- **范围**：`capx/envs/trial.py` 的 multi-turn 决策路径 + `capx/utils/launch_utils.py` 的 VDM prompt 构造。

## 2.（task-setting 相关）多视角 VDM

- **背景**：单一固定相机视角常看不清"是否真的抬起"这类细微变化，曾导致 VDM 误判。
- **现状**：当前 cube_lifting 等任务**未启用** wrist/第二相机（config 无 `use_wrist_camera`，web runner 也只用单视角 before/after），因此**不做多视角**——这是 task setting 决定的。
- **触发条件**：仅当某任务的 setting 本身提供 wrist/多相机时，才考虑把多视角喂给 VDM 增强判断；否则不适用。
