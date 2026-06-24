# Coding Expectation —— 每日开工须知

> 每次开始任务先读这份。它告诉你：我们在做什么、开工先问什么、按什么规则和风格推进。

## 0. 我们在做什么（self-evolve 的当前形态）

在 cap-x 之上做 **self-evolve**。当前阶段用 **dogfooding** 的方式推进：**亲手用 cap-x 暴露的
service / skill + 环境里可用的包，为 LIBERO PRO 的某个 task 编写一套能解决问题的代码**，借此
**发现 cap-x 缺什么**（缺的权限 / service，以及缺的 skill）。产物沉淀在 `capx-se/<task>/`
（每个 task 一个目录：解法代码 + runner + `GAPS.md` / `DISCUSSION.md` / `CONTINUE.md` 续接手记）。

## 1. 开工第一步：启动所有 service（除 openrouter）

**写任何 task 代码之前，先把 cap-x 的默认 service 全部起起来。** 这些是解题代码会调用的
**感知 + 规划工具**（SAM3 分割 / Contact-GraspNet 抓取 / PyRoKi IK·运动 / Molmo point-prompt），
缺一个就会让 skill 在中途报错。**唯一不在默认启动里的是 `openrouter`**（那是 agent 自己的 LLM
端点，单独起）。一条命令搞定（幂等，已在跑的会自动跳过）：

```bash
bash scripts/start_capx_services.sh           # 起全部，等就绪
bash scripts/start_capx_services.sh --status  # 只查 up/down，不启动
```

- 端口：SAM3 `8114` / GraspNet `8115` / PyRoKi `8116` / Molmo `8122`。日志在 `logs/services/`。
- 主 venv `.venv/bin/python` 起 SAM3 / GraspNet / PyRoKi。
- **Molmo（vLLM）**：跑在**专用 env `.venv-molmo`**（vLLM 依赖重、且与 cap-x 的 robosuite /
  transformers pin 冲突，所以单独建 env；molmo.py 走 HTTP，与解题 env 完全解耦）。脚本自动优先用
  `.venv-molmo` serve `allenai/Molmo2-8B`。没装时会 WARN 跳过，建 env：
  `uv venv .venv-molmo --python 3.12 && uv pip install --python .venv-molmo vllm==0.15.0`。
  缺它时 `point_prompt_molmo` 不可用，但 SAM3 仍覆盖分割需求。

## 2. 开工第二步：问我这次探索哪个 task

每次开始，先问用户「这次用哪个 task 来探索？」。**可探索的任务池 = LIBERO-PRO 的全部 dev 任务**，
即下面 4 个 dev suite（沿用 `track_judge` 的 dev 划分；排除两个 held-out 的 `libero_spatial_*`），
每个 suite 含 10 个 task（`task_id` 0–9）：

- **`libero_object_swap`**  —— object 套件的位置扰动（10 tasks）
- **`libero_object_task`**  —— object 套件的任务扰动（10 tasks）
- **`libero_goal_swap`**    —— goal 套件的位置扰动（10 tasks）
- **`libero_goal_task`**    —— goal 套件的任务扰动（10 tasks）

**held-out（不在 dev 池、勿当探索目标）**：`libero_spatial_swap`、`libero_spatial_task`。

> 当前已有续接的探索：`open_drawer`（`capx-se/open_drawer/`，对应 `libero_goal/task0` 的开抽屉）。
> 新任务用 `<suite>_<task_id>` 命名其 `capx-se/` 目录（如 `capx-se/libero_goal_task_3/`）。

列某个 suite 全部 task 的 language（确认 task_id 对应的指令）：
```bash
source .venv-libero/bin/activate
python -c "
from libero import benchmark
suite = benchmark.get_benchmark_dict()['libero_goal_task']()
for i in range(suite.n_tasks):
    print(f'  [{i}] {suite.get_task(i).language}')
"
```

用户选定后，进对应 `capx-se/<task>/` 目录，先读该目录的 `CONTINUE.md`（续接现状）再动手。

## 3. 硬性要求（不可违反）

- **解题路径不准读 simulator 特权状态**：物体/关节真值这种真机拿不到的，一律用 **sensing 估计**
  （agentview / wrist RGB-D + proprioception）。特权读取**只允许**在 runner 里做测量/打分，
  **绝不**进解法。
- **success 必要但不充分，还必须 SAFE**：不仅要过 benchmark 判定，还**不能碰撞/扰动其它物体**；
  runner 要同时报告 task success **和** 物体扰动量。**安全优先于成功** —— 宁可安全地失败，
  也不要不安全地“成功”（不要硬闯）。
- **不准 overfitting 单个 task / 单个 seed**：要通用解法 —— 闭环依赖感知反馈，别堆针对单场景
  调出来的魔数；要跨 seed / 场景鲁棒。
- **允许且鼓励多步交互**：先执行一段 → 观察当前环境状态 → 再写/跑下一段（闭环，而非一次性开环
  脚本）。**wrist（eye-in-hand）相机可用** —— 到 pre-grasp 后用它做近距离重检测（agentview 单独
  太粗，不足以稳定 seat）。
- **代码组织**：结构清晰、模块化但不过度模块化；**单文件不超过 500 行**；可多文件、不限一个。

## 4. ENV / 硬件

- 主环境 `.venv/bin/python`；部分依赖 `.venv-libero/bin/python`（LIBERO / 运动规划走这个）。
- web-ui 构建用 `~/.capx_nodeenv`。
- 机器有 **2×50GB GPU + 大内存**：默认 service 一键全开见 §1（`scripts/start_capx_services.sh`），
  还能随意再起别的（owlvit / sam2 / curobo 等），只要能解决问题任何模型都可以试。
  默认端口：SAM3 `:8114`、graspnet `:8115`、pyroki `:8116`、molmo `:8122`。
- 跑 LIBERO 一般要 `MUJOCO_GL=egl HF_HUB_OFFLINE=1`。

## 5. 流程性要求

1. **自动修改**：直接动手，不用每步确认。
2. **逐轮 debug 和验证**：拿不到特权信号就靠 sensing + runner 里的 ground-truth 测量来对照调试。
3. **每个验证通过的功能就 commit 落锚**（context 压缩 / session 切换后能从最近 commit 干净续上）。
   commit message **开头带 `[auto]`**（旧的用 `[tmp]`，历史 commit 不动）；**不**加 Co-Authored-By、**不**把 Claude 列为 contributor。
4. **随手保持 `capx-se/<task>/CONTINUE.md` 可接续**：每推进一步就更新「这次改了什么 / 还想改什么 /
   怎么跑」。中断在哪都能无缝续上。

## 6. 路标

- **当前任务续接**：`capx-se/<task>/CONTINUE.md`
- **该任务发现的 cap-x 缺口**：`capx-se/<task>/GAPS.md`、`DISCUSSION.md`
- 历史背景（旧 self-evolve pipeline 设计，可能部分仍参考）：`docs-se/`、`GOAL.md`（已逐步过时，以本文件为准）。
