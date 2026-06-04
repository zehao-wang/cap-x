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

---

## 旁路任务：human feedback → Feedback Postprocessor handoff（docs-se/03）落盘

> 用户诉求：debug 一条 interactive 真机 log 是否存够给下游 postprocessor agent 用的内容，
> 并让 human feedback 输出"易被 postprocessor 直接消费"。
> 查证 `logs/piper_interactive_20260602_181547/trial_01`：raw feedback 只埋在 `feedback_distill`
> 调用的 prompt 里、人的 Finish 成功信号根本没落盘（events 末尾是模型 FINISH，目录名还是
> headless 口径 reward_0.000）。→ 补两件事。

### ✅ 已完成
- `capx/utils/trace_logger.py`：
  - `log_human(kind="feedback"|"finish", text, turn)`：把人的动作记成一等 event（events.jsonl）+
    渲染进 trace.md；NullTraceLogger 加 no-op。
  - `write_handoff(...)`：trial 根写 `postprocess_handoff.json`（success-log schema 草稿，对齐
    docs-se/storage.md）：task / settings / final_code(人确认成功码，未通用化) /
    chat_history(跨 attempt 线性重建、带 0-based `index` 供 `[#idx]` 溯源、含 verbatim 人 feedback) /
    datetime + 显式 `success{signal,attempt}` + 便捷 `human_feedback` 列表。
  - `_build_chat_history`：读各 `attempt_NN/{events,llm_trace}.jsonl`，assistant 取 llm_trace 全文
    （非 240 预览），tool/human 按时序并入。
- `capx/web/async_trial_runner.py`：
  - 收到 feedback → `trace.log_human(kind="feedback", ...)`（落在它评判的 attempt 上）。
  - 人点 Finish → `trace.log_human(kind="finish")`（成功信号显式落盘）。
  - 成功保存段：`if human_finished:` 调 `trace.write_handoff(task=task_description,
    settings={model,env_config,visual flags}, final_code=final_code)`。

### 验证
- [x] 静态：两文件 py_compile。
- [x] 仿真：copy 旧 trial_01 + 注入合成 human 事件 → write_handoff 产出 35 条 chat_history，
      #17 human_feedback(中文 verbatim)、#34 human_finish、success.attempt=1 正确推断、
      assistant #2 为 401 字符全码（非预览）。旧 log 未被触碰。
- [ ] **live-gated**：下次真机交互跑通一次"人确认成功"，确认 trial 根真出 `postprocess_handoff.json`
      且 chat_history 含 verbatim feedback；通过再标 ✅。
- [ ] （可选）下游 postprocessor agent 真消费此 handoff（docs-se/03 仍 🔲 待实现）。

### ✅ 已完成（接续）：handoff 契约文档 + 模块化 debug 接口
- docs：`storage.md` 补 **postprocess handoff schema**（live-loop→postprocessor 输入契约）；
  `03-feedback-postprocessor.md` 补「输入契约」节；新增 `docs-se/debugging.md` 并挂进 README 文档地图。
- code：`capx/self_evolve/handoff.py`（`load_handoff`/`Handoff.postprocessor_kwargs`/`find_handoffs`，
  契约校验=「log 落盘对不对」判据）；`capx/self_evolve/debug.py`（CLI 三子命令 + 可注入
  `make_logged_query_fn`/`auto_accept_judge`/`reject_judge`/`interactive_judge`）；`__init__` 导出。
  - `debug handoff <trial>`：①feedback log 落盘检查（无 LLM）。
  - `debug postprocessor <trial>`：③读 handoff→rewrite+distill，dry-run 写临时 mem；`--skip-rewrite` 隔离 distill。
  - `debug planner --mem --tools-only`：④先打印 History Reader 工具暴露的 pool 视图，再可跑 proposer↔reviewer。
- 验证：`tests/test_debug_harness.py`（7 passed）；自演化全测 72 passed；三命令对真实/seed 数据手验通过。

### 待办（live-gated / 后续）
- [ ] 真机跑通一次成功 → 用 `debug handoff` 验证真出 handoff（接上一节 live-gated 项）。
- [ ] postprocessor/planner 用真 LLM（openrouter）跑一遍，肉眼核对 agent 产出。
- [ ] ⑤Evaluator/⑥Heartbeat 的孤立驱动暂未进 debug CLI（更重，需 sim/cron）。

### ✅ 已完成（接续）：三层记忆 · func_candidate_pool 进 git（中期记忆）
- 决策（用户）：`func_candidate_pool` = 中期记忆，进 git，随仓库流到另一集群做大规模 sim evaluation；
  `history_pool`(+`.processed_history`) 维持短期、本地 gitignored；长期库仍是 `capx/` 代码走 PR。
- `.gitignore`：`mem` → `/mem/*` + `!/mem/func_candidate_pool/`（check-ignore 验证：history_pool/cursor
  IGNORED、func_candidate_pool/* TRACKED）。
- 新增 `mem/func_candidate_pool/README.md`（占位 + 记录中期记忆角色；不被 `list_candidate_names` 误当候选，
  其按 `*.stats.json` 识别，已验证 list 仍为空）。
- docs：`storage.md` 树注明三层 + git；`concepts.md §3` Memory 改为短/中/长三层。

### ✅ 已完成（接续）：③ refine 升级为 (A)/(B) 判别 + 感知重推 + api_reference 接线
- 用户定调：③ 直接写完整原理化实现（含新机制）；不是所有超参都该重推——(A) 通用超参（~2cm grasp
  z-margin、沿 −approach 的 pre-grasp 回撤）保留成命名超参；(B) 场景专属（手调 waypoint/轨迹）按推断
  的约束用感知 API 在运行时重推。④ 再去重/固化可复用机制。
- `feedback_postprocessor.py`：重写 `build_generalize_rewrite_prompt`（教 (A)/(B) 判别、re-derive、
  允许新机制、env-in-loop 判对）；新增 `_human_feedback_from_chat`（从 chat 的 human_feedback 轮取
  verbatim 指导）；`generalize_by_rewrite`/`run_feedback_postprocessor` 透传 `human_feedback`+`api_reference`。
- `handoff.py`：Handoff 加 `api_reference`，`load_handoff` 缺字段时从 trial 的 `attempt_*/llm_trace.jsonl`
  回收（含 "APIs:" 的最长输入文本）→ 旧 handoff 也能用；`postprocessor_kwargs` 带上它。
- `trace_logger.write_handoff` + runner：未来 run 直接落 `api_reference`（runner 从 `clean_base_prompt` 抽）。
- `debug.py`：postprocessor 打印 api_reference 状态 + 当 `--judge auto` 且有 rewrite 时警告"离线不算验证"。
- 验证：75 passed（含新增 3 个 (A)/(B) prompt 测试）；旧 handoff 回收 api_reference=8082 chars（含
  get_scene_view/refresh_point_clouds/get_object_pose），feedback 抽到 2 条。
- docs：`03` 做法节重写（(A)/(B) + ③↔④ 边界 + env-in-loop）；`storage.md` handoff schema 加 `api_reference`。
- 待办：用真 LLM 跑一次升级后的 `debug postprocessor`，肉眼看 (B) 有没有真的改成基于 point cloud 的约束重推。

### 接续：refine 深度定档 + prompt 收紧（用户验证后定调）
- 真跑 `postprocessor_20260604-170049` 验证：(A)/(B) 判别确实启动（call#1 就带 (A)/(B) 注释），
  (A) z-margin 留成命名超参 ✓；(B) waypoint 高度改成 `obj_height(get_object_pose bbox)+命名 clearance`
  → **只到"目标物体自适应"**，没到"基于场景障碍/point cloud 的约束轨迹"；clearance 量仍是命名常量；
  XY pullback 仍假设 +x；auto-judge 没验证；digest 漏报这些 fragility。
- 决策（用户）：**当前深度先接受**——无场景 point cloud 时这种 refine 难测有效性，完整的 point-cloud
  约束轨迹采样器**留给长期更新**（将来作为可复用 skill / ④ candidate）。不强推。
- 用户反馈：prompt 加太多。已收紧 `build_generalize_rewrite_prompt` 的 system（~230→136 词），
  保 (A)/(B) 意思去冗余；测试仍 10 passed。
