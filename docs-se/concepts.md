# Concepts（术语表）

> **共享契约** — 被所有功能文档引用。**术语必须全文一致**（见每条末尾的命名约定）。

1. **skill**（openclaw 语境）：一段可更新的工作流，关注**多个 task 之间如何协作**。
   ⚠️ 与 **`skill_library`**（条目 3、[storage.md](storage.md)）是**两个不同概念**，勿混淆：
   本条 "skill" 是 openclaw 式高层工作流（如未来处理 rgbd demo 的 skill，见条目 7）；
   `skill_library` 是长期记忆里由 primitives 组成的通用函数库。

2. **system prompt**：每次 LLM 调用都固定出现的内容，规定必须考虑的事情和固定流程。

3. **Memory**：分短期、中期、长期（按生命周期 / 是否进 git 分层，见 [storage.md](storage.md)）。
   - **短期记忆 = `history_pool`**：人机交互中**成功**积累下来的经验（哪些能做 /
     不能做、code 层面哪些与人协作的更改让任务成功了）。**只有成功的 history 进入短期
     记忆**，且**只被 Library Management 消费**。本地运行时态，**gitignored**。
   - **中期记忆 = `func_candidate_pool`**：Update Planner 提出、尚待 benchmark 验证的候选函数
     + 累积统计。**进 git 跟踪**，使候选随仓库流到**另一个集群做大规模 simulation evaluation**
     （Benchmark Evaluator 在那边累加 stats）。
   - **长期记忆**：经过完整 benchmark evaluation 验证、确实对 task success 有贡献的
     coding 层产物（code / config / hyper-param），落在 `primitives`、`skill_library`、
     `atomic_task_library` 中（走 PR 固化）。
   - 存储布局与 schema 见 [storage.md](storage.md)。

4. **Heartbeat**：定时唤起一些 daily task 或长期 plan 的 task，是推动模型演进的机制
   （参考 openclaw 实现）。长期目标与 daily task 见 [06-heartbeat-cron.md](06-heartbeat-cron.md)。

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
     相关的 margin 作为 grasp atomic task 的 config 字段（见
     [03-feedback-postprocessor.md](03-feedback-postprocessor.md)）。
   - **命名约定**：全文统一用 **atomic task library**。（旧文档中出现过的 "subtask
     library" 即指此物，已废弃该叫法。）

9. **Feedback Postprocessor**：当一次任务**依赖 human feedback 给的额外信息**才成功后启动
   的流程，负责把代码**通用化 / 超参化**，去除对这次一次性额外信息的直接依赖。详见
   [03-feedback-postprocessor.md](03-feedback-postprocessor.md)。

10. **VDM**：见 [README.md](README.md) §1——cap-x 内用于自动评测的 agent。Benchmark
    Evaluator 中继续用它做自动成功判定。
