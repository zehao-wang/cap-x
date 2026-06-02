# 长期目标 / Long-term Goals

> 本文件是 **self-evolve agent 的完整期待 + 进度总账**：记录我们要做成什么、当前到哪了、
> 接下来分哪些模块推进。设计真相在 `docs-se/`（本文件只做"任务分解 + 状态"，不重复设计细节）。
>
> **怎么用**（配合 `AGENT.md`）：
> 1. 读本文件 → 看「Pipeline 状态」挑出本次要推进的模块；
> 2. 把该模块拆成可执行步骤写进 `Short-Term-GOAL.md`（临时 memory）；
> 3. 分模块 coding；该模块做完 → 在本文件对应行标 ✅、清空 `Short-Term-GOAL.md`。
>
> 设计阅读顺序：`docs-se/README.md` → `concepts.md` + `storage.md`（术语 + 数据契约）→ 具体模块文档。

---

## 0. 北极星 / Long-term vision

来自 `docs-se/06-heartbeat-cron.md`：

1. **提升 benchmark 上的效果**；
2. **在人机交互的 task 上更快、更容易成功**。

机制：把*人确认成功*的交互，经「通用化 → 蒸馏 → 候选 → benchmark 自动验证 → 人批准」闭环，
固化成**可验证、需人批准、可终身更新**的长期能力。整体闭环图见 `docs-se/00-overview-diagram.md`。

---

## 1. 待实现系统（cap-x-se）的设计硬约束

> 这些是 **cap-x-se agent 运行时必须满足**的流程/设计约束（用户拍板，易在实现中被忽略），
> 不是对"在本仓库编辑/commit"的限制。细节见各模块文档。

- **success signal 只能由人给**：VDM 不在 live loop 判成功；VDM 仅服务 Benchmark Evaluator 的自动评测。
- **更新长期 library 走 Pull Request**：cap-x-se 与人在 manipulation task 交互后要更新 library 时，把改动
  落成一个 PR（附报告）**等用户 merge**（而非弹聊天框）。Evaluator 不直接写长期库 / 不直接删 candidate；
  **用户 merge = 批准、close = 否决**。
- **遗弃不记 negative**：只按 `positive` 阈值固化；只对「久占 pool、从没被用、且无人引用」的候选遗弃。
- **术语统一**：用 **atomic task library**（旧称 "subtask library" 已废弃）；atomic task 不绑具体物体/任务。
- **路径**：长期库固化到 `capx/skill_library/`、`capx/atomic_task_library/`；**不**直接改 `capx/skills/library.py`；
  pool 在仓库根 `mem/`。
- **暂不做**：Memory Compact；rgbd video feedback 为未来。

---

## 2. Pipeline 状态总览

| 阶段 | 文档 | 状态 | 落地位置 / 备注 |
| --- | --- | --- | --- |
| ① Live Loop（human-in-the-loop 主循环） | `01-interactive-loop.md` | ✅ | `capx/web/async_trial_runner.py`（已模块化出 `trial_support`/`vdm_feedback`/`trial_artifacts`/`reset_wizard`/`session_manager`）+ `capx/envs/base.py` |
| ② Visualization（viser 回放） | `02-visualization.md` | ✅ | `capx/utils/viser_history.py`、`viser_history_io.py`、`viser_playback_panel.py` |
| ③ Feedback Postprocessor + Experience Distill | `03-feedback-postprocessor.md` | 🟡 | 核心已实现+单测：`capx/self_evolve/feedback_postprocessor.py`；**剩 live-loop 接线**（async_trial_runner `human_finished` 分支 + 环境交互式 generalize）为 sim-gated，暂挂等真人在 web/sim 联调 |
| ④ Update Planner | `04-update-planner.md` | ✅ | `capx/self_evolve/{history_reader,proposal,update_planner}.py` + `tests/test_update_planner.py`（注入 query_fn，scripted 全程单测） |
| ⑤ Benchmark Evaluator | `05-benchmark-evaluator.md` | 🟡 | 核心已实现+单测：`capx/self_evolve/{benchmark_eval,library_pr}.py` + `tests/test_benchmark_evaluator.py`（注入 `eval_fn`，stats 累加/引用扫描/promote-abandon-keep/报告/PR-plan 全程单测）；**剩**真正的 sim `eval_fn`（注入 candidate 跑 benchmark + VDM）+ `create_library_pr` 的 gh/push 链路为 live-gated（本机无 gh），等 ⑥ cron 接线时联调 |
| ⑥ Heartbeat / Cron | `06-heartbeat-cron.md` | 🟡 | 核心已实现+单测：`capx/self_evolve/scheduler.py`（`CronSpec` 5字段匹配 / `select_daily_tasks` 低正确率选择 / `Heartbeat.tick` 分钟级去重 / `make_evaluator_job` 夜间触发⑤(空池no-op) / `make_daily_task_job` 派发交互循环）+ `tests/test_scheduler.py`；config 加 `heartbeat.*` 组。**剩** `run_forever` 长驻 + 真正 accuracy_provider / interactive_loop / sim eval_fn 接线为 live-gated |
| ⓪ 共享存储脚手架（`mem/` schema + config） | `storage.md` / `config.md` | ✅ | `capx/self_evolve/`（`config.py` / `schemas.py` / `storage.py`）+ `tests/test_self_evolve_storage.py` |

> ⚠️ 现状提醒：⓪ 已落地——`mem/` 的读写契约在 `capx/self_evolve/`（`MemStore` 按需 lazily 建
> `history_pool/` `func_candidate_pool/` + `.processed_history`），但 pool 目录本身要等 ③ 真正写入才出现；
> 长期库 `capx/skill_library/` `capx/atomic_task_library/` 两个模块仍不存在（等 ⑤ 批准后才建）。
> 代码里已有的 `feedback_distill` phase 是 live loop 内折叠 operator guidance 的，**不是** module ③
> 的 Experience Distill —— 两者别混淆。

---

## 3. 待实现任务分解（按依赖顺序）

> 依赖链：⓪ → ③ → ④ → ⑤ → ⑥。每条标注 *读/写契约*、*交付物*、*关键约束*、*完成判据*。

### ⓪ 共享存储脚手架（前置）

- **为什么先做**：③④⑤ 都按 `storage.md` 的 schema 读写 `mem/`，先把数据契约落成代码可省去后续返工。
- **交付物**：
  - `mem/` 目录布局 + 读写辅助（`history_pool/`、`func_candidate_pool/`、`.processed_history` 游标）。
  - success log schema / digest schema / candidate stats schema 的序列化与校验（schema 见 `storage.md`）。
  - 一处集中的 config（`experience_distill.*` / `update_planner.*` / `benchmark_eval.*`，默认值见 `config.md`）。
- **完成判据**：能用辅助函数读写三种 schema 的样例文件并通过校验；config 项可被各模块引用。

### ③ Feedback Postprocessor + Experience Distill

- **依赖**：① 产出的*人确认成功* trial；⓪ 的 history schema。
- **读/写**：成功 trial → `mem/history_pool/<task>__<ts>.json`（通用化后的 `final_code` + 全文）
  **和** `<task>__<ts>.digest.md`（蒸出的 sourced digest）。
- **交付物**：
  1. **Generalize-by-rewrite**：多轮 code rewrite 去掉对一次性 feedback 的直接依赖（如把硬编码
     `z += 0.03` 升成 grasp atomic task config 的超参）；**rewrite 仍与环境实际交互**，**每轮只由人判对错**。
  2. **Experience Distill**（定稿后、入 pool 前跑一次）：debugger 式问固定问题（KEY STRATEGY /
     REUSABLE PATTERN / KEY HYPER-PARAMS+来历 / FRAGILITY），强制限长 `experience_distill.max_words`
     （默认 200），每条论断带 `[#message_index]` 回溯标注。
- **关键约束**：只有成功的 trial 进 pool；digest 是有损入口视图，真相在 `.json` 全文。
- **完成判据**：一次人确认成功的 web trial 跑完后，`history_pool/` 落出成对的 `.json` + `.digest.md`，
  digest 限长且每条带 `[#idx]`，能 drill 回全文对应轮次。

### ④ Update Planner（Library Management 前半段）

- **依赖**：③ 的 `history_pool`（默认读 digest）+ ⓪ 的 candidate stats schema。
- **触发**：`history_pool` 未处理数 ≥ `update_planner.trigger_history_count`（默认 5）自动跑；
  用 `mem/.processed_history` 记游标，只处理增量。
- **交付物**：
  1. **History Reader 只读工具集**：`list_unprocessed` / `read_digest` / `read_history(field,offset,limit)` /
     `grep_history` / `read_library` / `propose`（终结工具）。**digest 优先、按需 drill**，单次调用读取
     硬上限 `update_planner.max_read_iterations`（默认 20）。cap-x 自有实现，不依赖外部 `adb` CLI。
  2. **LLM-1 提议 → LLM-2 审核** 双 LLM 循环：审核查 `source_history` 引用属实 / 去重 / 无 bug /
     够 general / 粒度 ≤ atomic（拒 `put_apple()` 这类绑物体的）；pass 由 **LLM-2 写入**
     `func_candidate_pool`（含 `*.stats.json`），不 pass 带 feedback 让 LLM-1 revise，上限
     `update_planner.max_revise_iterations`（默认 5）超限强制定稿。
  3. 处理完这批 → 标记已处理（history 文件保留）。
- **交付物结构**：`propose` payload = `{candidates:[{func_name,target_library,code,source_history,rationale}]}`。
- **完成判据**：攒够 ≥5 条 history 后自动触发，落出 `func_candidate_pool/<func>.py` + `.stats.json`，
  `source_history` 指向真实 id，`.processed_history` 正确推进。

### ⑤ Benchmark Evaluator（Library Management 后半段）

- **依赖**：④ 的 `func_candidate_pool`；`integration.md` 的注入点；VDM。
- **触发**：夜间 cron 一次（⑥ 调度）或人工唤醒；仅 pool 非空时跑。
- **交付物**：
  1. **注入**：用 `SkillLibrary.get_skill_docs()` / `inject_into_namespace()` 把 candidate 作为**可选工具**
     加入，在 sim 里跑 benchmark（类似 `scripts/run_agent0_qwen36_{robosuite,libero}.sh`），**VDM 自动判成功**。
  2. **统计累积**（跟随 candidate，不记 negative）：`positive` / `used_runs` / `eval_runs`。
  3. **决策 + 报告**：promote（`positive ≥ promote_threshold`(10) 且 `eval_runs ≥ min_samples`(5)）/
     abandon（`eval_runs > max_idle_evals`(50) 且 `used_runs==0` 且无人引用，引用关系删除前动态扫描）/
     keep；固化函数 docstring 记更新日期。生成简单报告。
  4. **批准 = PR**：把本轮 promote（写 `capx/skill_library/` / `capx/atomic_task_library/`）与 abandon
     （删 `func_candidate_pool` 候选）的改动落成**一个 PR**，报告作为 PR 描述。**Evaluator 自己绝不直接
     改长期库 / 删候选**——一切等用户 **merge（批准）/ close（否决）**。
- **关键约束**：任何 promote/delete 只能经 PR、由用户 merge 后生效；报告只提议不动手。
- **完成判据**：对非空 pool 跑一轮后 stats 正确累加、产出建议报告并开出一个含改动的 PR；
  只有 PR 被 merge 后 `capx/skill_library/` / `capx/atomic_task_library/` 才更新、候选才从 pool 删除。

### ⑥ Heartbeat / Cron（调度层）

- **依赖**：① live loop（产新成功 history）；⑤ Evaluator。
- **交付物**：
  1. **Heartbeat**：唤起 daily task —— 把*正确率低*的 task 拎给人 feedback（走 ①），用新成功 history 推动 library 更新。
  2. **Cron**：固定时刻触发 heartbeat；典型为夜间固定时间触发 Benchmark Evaluator（参考 openclaw 实现）。
- **完成判据**：cron 能在设定时刻自动触发 Evaluator；heartbeat 能挑低分 task 派给交互循环。

---

## 4. Backlog / 暂缓（不在主线，触发条件满足再做）

- **用户搁置的候选目标在 `TASK_TMP_ASIDE.md`** —— 那是用户自己维护的「也许将来做」清单（当前含：
  headless benchmark 同步 web 的 VDM/错误处理改进、task-setting 相关的多视角 VDM）。每条带触发条件，
  满足后再升入上面的主线 §3。本文件不重复其内容，避免两处维护。
- **来自设计的未来项**（`docs-se/concepts.md`）：Memory Compact（§6，目前不做）、rgbd video feedback
  （§7，human feedback 当前仅 text；未来支持 rgbd demo 视频 → 专门 skill 处理）。

---

## 5. Real-robot 实验设置 / Real Robot Setup

> 真机实验是 self-evolve 主线（§2–§4）之外的**独立链路**：在 AgileX Piper 真机上跑 agent0 交互闭环。
> 本节只记**实验设置 + 硬约束**；连接/调试的逐轮进展放 `Short-Term-GOAL.md`。

### 硬件

- **机械臂**：AgileX Piper 6-DOF，CAN 总线驱动（`piper_sdk`）。
- **场景相机**：**ZED 2i** → `obs["robot0_robotview"]`（RGB + depth + 内参 + 外参）。normal flow 下 RGB+depth+内参
  **只来自独立 ZED 服务**（见下），相机参数归服务侧，不在 task yaml 配。外参（base 系位姿）由 cap-x 标定文件
  `env_configs/real/piper_zed_extrinsics.yaml` 提供，不来自服务。
- **腕部相机（可选）**：RealSense D435* → `obs["robot0_eye_in_hand"]`（cap-x 本地直连）。
- 落地配置：`env_configs/real/piper_real.yaml`（本地）/ `piper_real_service.yaml`（机械臂状态走服务）。

### ZED 相机 = 独立服务（设计决定，用户拍板）

- ZED 相机**单独启动一个服务**，对外返回 **RGB + depth + 内参 + timestamp**（**外参不归服务**）。
- 该服务**独占相机**，并在**服务侧**完成所有深度计算（如基于 ZED 左右目的 TRI-Stereo 学习深度）。
- **cap-x 只负责读**：经固定协议（UDS）取 RGB+depth 写进 obs，**不在 cap-x 做任何多余处理**
  （不跑 tri-stereo / 不算 SDK 深度 / 不加深度模型依赖）。服务实现契约见
  `capx/envs/simulators/piper/ZED_SERVICE_REQUIREMENTS.md`。
- **就绪 + 访问约定**：服务启动时自己跑一次读取自检，**通过才算开启成功**（⇒ socket 可连即保证有合法数据）；
  cap-x 侧**不设放弃超时**——服务不可达时**心跳等待直到恢复**，并在屏幕告警（请检查相机服务状态 + 当前失败原因）。
- 理由：① 相机单 owner（ZED 只能被一个进程 open）；② 依赖隔离——pyzed（numpy<2）、深度模型的
  onnxruntime/权重全留在服务侧，cap-x 主环境保持干净；③ 与已有 `piper_state_service`（机械臂状态走
  websocket 服务、cap-x 当客户端）一致的「硬件跑服务、cap-x 当瘦客户端」模式。

### 状态

- 🟡 **ZED 相机服务**：按 `ZED_SERVICE_REQUIREMENTS.md` 实现于 `scripts_realbot/zed_service/`
  （`zed_depth_service.py` + `run_zed_service.sh`，用 raiden venv 跑 pyzed + TRI-Stereo）。线格式 v1 +
  就绪自检 + 干净退出已写好；loopback（真 client）+ 真 TRI-Stereo backend 加载均已验证。**卡在缺真机/显示器**
  —— 还差真开相机常驻跑一次。
- ✅ **cap-x 侧瘦客户端 + 接线**：只读、UDS、心跳等待；经 config `piper_zed_source: service|bridge` 选择。
- ✅ **normal flow 收口为 service-only**（commit `4c7a62f`）：`piper_real.yaml` 已设 `piper_zed_source: service`
  + `piper_zed_service_socket`，并**删掉 yaml 控相机参数的能力**（fps/分辨率/depth_mode/曝光增益）——这些归服务侧 owner，
  cap-x 不再能从 task yaml 调相机。端到端拆掉 yaml→bridge 调参链路（base.py / piper_real.py / setup.py /
  launch_piper_state_service.py），state service 也跟 yaml 走 service。`_Zed2iBridge` 与 bridge 分支仅保留给
  **标定脚本**（直接构造、读 `PIPER_ZED_*` env）。（细节见 `Short-Term-GOAL.md`）
- 🔲 **Piper 实机交互闭环跑通**：本地实机 + OpenRouter Gemini（**不用 qwen3.6**）。
- ✅ **已撤销**「在 cap-x 内跑 TRI-Stereo」的旧做法，深度全部移到服务侧（cap-x 不持有任何深度模型/依赖）。
