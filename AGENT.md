# Coding Expectation —— 每日开工须知

> 每次开始任务先读这份。它告诉你：我们在做什么、开工先问什么、按什么规则和风格推进。

## 0. 我们在做什么（self-evolve 的当前形态）

在 cap-x 之上做 **self-evolve**。当前阶段用 **dogfooding** 的方式推进：**亲手用 cap-x 暴露的
service / skill + 环境里可用的包，为 LIBERO PRO 的某个 task 编写一套能解决问题的代码**，借此
**发现 cap-x 缺什么**（缺的权限 / service，以及缺的 skill）。产物沉淀在 `capx-se/<task>/`
（每个 task 一个目录：解法代码 + runner + `GAPS.md` / `DISCUSSION.md` / `CONTINUE.md` 续接手记）。

## 1. 开工第一步：问我这次探索哪个 task

每次开始，先问用户「这次用哪个 task 来探索？」并列出候选（至少包含已有的）：
- **`open_drawer`** —— 当前主力探索任务（`capx-se/open_drawer/`，先读其 `CONTINUE.md`）。
- 之后可能新增其它 LIBERO PRO task。

用户选定后，进对应 `capx-se/<task>/` 目录，先读该目录的 `CONTINUE.md`（续接现状）再动手。

## 2. 硬性要求（不可违反）

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

## 3. ENV / 硬件

- 主环境 `.venv/bin/python`；部分依赖 `.venv-libero/bin/python`（LIBERO / 运动规划走这个）。
- web-ui 构建用 `~/.capx_nodeenv`。
- 机器有 **2×50GB GPU + 大内存**：可随意起多个 service（SAM3 / graspnet / pyroki 等），
  只要能解决问题，任何模型都可以试。常用：SAM3 `:8114`、graspnet `:8115`、pyroki `:8116`。
- 跑 LIBERO 一般要 `MUJOCO_GL=egl HF_HUB_OFFLINE=1`。

## 4. 流程性要求

1. **自动修改**：直接动手，不用每步确认。
2. **逐轮 debug 和验证**：拿不到特权信号就靠 sensing + runner 里的 ground-truth 测量来对照调试。
3. **每个验证通过的功能就 commit 落锚**（context 压缩 / session 切换后能从最近 commit 干净续上）。
   commit message **开头带 `[auto]`**（旧的用 `[tmp]`，历史 commit 不动）；**不**加 Co-Authored-By、**不**把 Claude 列为 contributor。
4. **随手保持 `capx-se/<task>/CONTINUE.md` 可接续**：每推进一步就更新「这次改了什么 / 还想改什么 /
   怎么跑」。中断在哪都能无缝续上。

## 5. 路标

- **当前任务续接**：`capx-se/<task>/CONTINUE.md`
- **该任务发现的 cap-x 缺口**：`capx-se/<task>/GAPS.md`、`DISCUSSION.md`
- 历史背景（旧 self-evolve pipeline 设计，可能部分仍参考）：`docs-se/`、`GOAL.md`（已逐步过时，以本文件为准）。
