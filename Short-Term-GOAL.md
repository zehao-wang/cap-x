# Short-Term-GOAL —— 本次任务暂存

> 临时 memory：只放「本次正在推进的那个模块」的可执行步骤与进展。
> 模块做完 → 在 `GOAL.md` 对应行标 ✅，然后清空本文件回到模板。

## 本次模块：piper interactive 真机两处交互修正（+ openrouter 启动收口，已完成）

真机 `bash scripts_realbot/run_agent0_piper_interactive.sh` 跑通后，用户反馈两个问题。

### ✅ 已完成（scripts 收口）
- `scripts_realbot/openrouter_service/run_openrouter_service.sh` + `README.md`：openrouter 启动单一真相；
  launcher 调用它，checklist 末尾给出手动启动路径。
- 交互 launcher CAN 默认 **`can0`**（机械臂在 can0；can1 是另一块适配器）；修 `PIPER_CAN_INTERFACE`
  误当通道名的 bug。（注：曾短暂改成 can1 对齐其它脚本，用户按真机实际改回 can0。）
- **修真 bug：launcher 只算 `CAN_CH` 做 preflight，从不 export `PIPER_CAN_CHANNEL`** → env 子进程读不到，
  回落 `piper_real.py` 代码默认 `can1` → 运行时 `ConnectionError ['can1']`。修：launcher `export
  PIPER_CAN_CHANNEL="${...:-can0}"` 让 preflight 与 env 同源；`piper_real.py:51` 代码默认也改 `can0`。
  待办：`piper_set_mode.py` / cam_calibration 几个标定脚本的 arm channel 默认仍是 can1（独立工具，未改，需要时再统一）。
- helper 等待 30s→60s。

### 🔲 问题 1：viser Camera View 不刷新 ZED RGBD
- 根因：`piper_real.yaml` 走 `piper_real_low_level`（非 state service），viser 面板只在
  `motion.py`（运动中）和 `reset()` 刷新；交互态大部分 idle，无后台循环 → 面板停在旧帧。
- 决策：env 存活期一直刷，后台线程 ~5Hz，`_update_from_hardware()` + `_update_viser_server()`。
  硬件读用 `_hw_lock` 与 motion 串行化（ZED socket / RealSense / CAN 非线程安全）。
- 步骤：
  - [x] `piper/setup.py _init_viser`：加 `_hw_lock` + `_live_preview_*`（在 `if not viser_debug` 前；`import threading`）。
  - [x] `piper_real.py _update_from_hardware`：整体 `with self._hw_lock:`。
  - [x] `piper/viser.py`：`start_live_preview()/stop_live_preview()/_live_preview_loop()`。
  - [x] `async_trial_runner.py`：reset 后 `low_level_env.start_live_preview()`；`finally` 里 `stop_live_preview()`。

### 🔲 问题 2：interactive 不应直接开始任务，init env 后第一步要输入任务
- 根因：reset 后无条件立刻 `_stream_query(INITIAL)`；问人只在模型 loop 结束后。
- 决策：复用聊天框 gating。reset 后进 `AWAITING_USER_INPUT`，提示「请输入本次 trial 的任务」，
  用户输入 → 注入 prompt → 才开始第一次生成。开关用 cfg `prompt_for_task`（piper `is_real_robot` 为 False，不可用）。
- 步骤：
  - [x] `tasks/base.py CodeExecEnvConfig`：加字段 `prompt_for_task: bool = False`（dataclass 不吃未知 kwargs）。
  - [x] `env_configs/real/piper_real.yaml`：cfg 下加 `prompt_for_task: true`。
  - [x] `async_trial_runner.py`：env ready 后、视觉反馈/VDM 之前，若 `cfg.prompt_for_task` 则
        gate 在 AWAITING_USER_INPUT，loop 等非空 payload（FINISH=结束 trial）→
        `_append_task_to_prompt(obs["full_prompt"], task)`（模块级 helper）。

### 🔲 问题 1b：web 代理连到错误/残留的 viser（用户：viser 没用对，换个不易冲突的 port）
- 根因：其它 env 用 `viser.ViserServer()` 默认 8080/8081，web `_find_viser_port` 只探 `[8080,8081]`；
  **唯独 piper `setup.py` 用 `CAPX_VISER_PORT` 默认 8201**，代理探不到 → 连到 8080 上残留 viser
  （或标定脚本默认也 8201，留下脏 viser）。
- 修：
  - [x] `capx/web/server.py _find_viser_port`：优先认 `CAPX_VISER_PORT`（高于 cache），再回退 8080/8081。
  - [x] `run_agent0_piper_interactive.sh`：`export CAPX_VISER_PORT=${CAPX_VISER_PORT:-8211}`（避开 8080/8201），
        env 与代理都用它；launch banner 显示 `viser: :8211`。
  - [x] `piper/io.py close()`：补 `stop_live_preview()` + `viser_server.stop()`，释放端口（__del__ 会调 close）。
  - 注意（未改）：runner 不在 trial 间 `env.close()`，同一进程内开第 2 个 trial 仍可能撞 8211→viser 自增 8212、
    代理停在旧的。重启 launcher（新进程）必干净。要多 trial 复用需加 per-trial env teardown（待定）。

### 🔲 问题 3：feedback 弹「No episode snapshot」Error + 无重摆场景暂停
- 根因：piper 没被认成真机（`is_real_robot` 靠 `return_to_rest_pose`，只有 franka 有）→ feedback 走
  sim 分支，无 `restore_state` → 弹 ErrorEvent 并直接回零重跑，人没机会重摆物体。
- 决策（用户）：认成真机走 guided 向导（同 franka）。
- 修：
  - [x] `piper/motion.py`：加 `return_to_rest_pose()`（`_set_gripper(1.0)` + `goto_home_blocking()`）。
  - [x] `piper/io.py`：加 `is_connected(max_age=2.0)`（best-effort 读 `GetArmJointMsgs().joint_state`）。
  - 效果：`is_real_robot` 对 piper 变 True → 初始 reset 与每次 feedback 都走 3 步向导
    （连接检查→自动回 rest pose 确认→确认场景已重摆），不再走 snapshot-error 分支。

### 验证
- [x] 静态：8 文件 py_compile；piper 暴露 is_connected+return_to_rest_pose（is_real_robot→True）；CodeExecEnvConfig 字段 + yaml cfg 键全合法（无未知键）；
      runner helper round-trip；piper env live-preview 方法可解析、`_init_viser` 设锁、`_update_from_hardware` 持锁；
      `viser.ViserServer.stop()` 存在；server.py / launcher 语法过。
- [ ] **真机（live-gated）**：跑 launcher 确认 (1) viser Camera View 实时跟动且是 ZED 画面（:8211）；
      (2) reset 完成后 UI 等输任务、输入后才出码。通过再 `[tmp]` commit。
