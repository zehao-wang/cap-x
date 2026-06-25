# open_drawer — 相比原始 cap-x，是哪些设计让我们做到 5/5

目标：`libero_goal/task0` 开中间抽屉，**只用 sensing（agentview/wrist RGB-D + 本体感觉），
解题路径不读 simulator 特权状态**。原始 cap-x 的 reduced-API 原语（`segment_sam3` /
`plan_grasp` / `solve_ik` / `goto_pose` / `move_to_joints` / 夹爪）单独用时：抽屉能被
"打开"，但会**撞翻前方物体**（bowl ~210 mm、plate ~40–60 mm），且对 seed/场景不鲁棒
（baseline 3/5）。最终结果：**官方 `run_robust.py` 冷启动 5/5 全开（qpos −0.160），
物体扰动 4–25 mm**。下面按对结果的贡献排序，列出我们在 cap-x 之上加的设计。

---

## 1. 决定性的一招：**确定性 IK 落座**（替掉随机 trajopt 落座）
- **原始 cap-x**：唯一的运动引擎是 HORL/pyroki **trajopt**（轨迹优化）。它对长程 transit 没问题。
- **限制**：把它用于最后 5 cm "把夹爪精确坐到细把手上"时，**落点 z 随 RNG 在 0.10–0.17 漂**
  （同一目标，warm planner 偶尔坐准、cold planner 经常坐高）。坐高 1–6 cm → 夹在把手上方 →
  抓空 → 拉不动抽屉。这就是 success 不稳定的根因（GAPS §G）。
- **我们的设计**：用 `solve_ik(bar_pose, quat)` 求一个**到把手位姿的关节解**，再沿
  **sensing 已验证为 clear 的 pre→bar 走廊**做**关节线性插值**执行。落点**可复现地又深又居中**
  （y≈−0.137、z≈0.11，实测 0.109/0.110/0.110，零散布）。冷启动每个 seed **第一次就坐准**。
- **可复用**：cap-x 缺一个"短程确定性落座"原语；trajopt 适合 transit，不适合精抓。我们用
  solve_ik + 线性插值 + 走廊净空校验把它补上了。

## 2. **从深度点云建碰撞世界** + HORL planner 接入（session 1 打的底）
- **原始 cap-x**：reduced-API 没有 collision-aware execution，原语直线走 → 撞物。
- **我们的设计**：把 agentview depth 反投影成点云，过滤出"把手前方的桌面物体"，喂给 HORL
  RRT+trajopt 做**球碰撞世界**（`horl_planner.py`）。所有 transit 绕开物体而不是犁过去。
  扰动从 ~210 mm 量级降到个位/十位 mm。

## 3. **安全门用 sensing，而不是 planner 的内部 cost**
- **原始 cap-x**：trajopt 的 collision cost 是唯一信号；但它**会给假阳性**——夹爪在比 plate 高
  7 cm 处掠过，cost 也能爆到 ~1e10（甚至 NaN），而真实扰动只有 ~3 mm。直接拿它当安全门会
  把本来安全的动作误杀（seed 1 旧版就这样 safe-abort）。
- **我们的设计**：安全/接受判据全部落在**可观测量**上——执行后用本体感觉读 TCP 是否真的坐到
  把手（`on_bar`）、夹爪闭合读数是否抓住（grip>0.05）、走廊净空由 sensing 点云算。
  "verify-by-result"，不信 trajopt 的 cost。

## 4. **带位姿校验的 descend-to-pre**（RRT 静默失败的兜底）
- **原始 cap-x**：靠近结构时 RRT 常**静默返回 not-planned**（"start tree could not be
  initialized"），调用方拿不到信号就从错误位置继续（卡在高 standoff → 坐高 → 抓空）。GAPS §H。
- **我们的设计**：RRT 重试 → 不行就 collision-aware trajopt 兜底 → **校验 TCP 真的到了 pre**
  再落座。这一步是让 IK 的 warm-start 干净、从而每次坐准的前提。

## 5. **撤退用 trajopt，不用 RRT**（修一个会卡死的 hang）
- **原始 cap-x**：拉开后机械臂停在抽屉口，RRT 把**起始**构型判成 in-collision 直接卡死
  整个 run（抽屉其实已经开了）。
- **我们的设计**：撤退改用 collision-aware trajopt——它接受这个起点，直接抬走。

## 6. **几何感知的抓取策略**（D-bracket / z-居中 / gap 进入）
- 通过 wrist 近距离重检测 + 读 LIBERO 资产，认清把手是**沿 X 的 D 形横杆**（与拉向 +Y 垂直）：
  所以"竖直方向坐准杆心"才是握牢的关键（之前的"打滑"其实是坐高了 1–3 cm 只夹到杆顶）。
- **gap 进入**：从 plate 近边和把手之间的**空档**降下去坐，而不是贴着把手高度从 plate 上方
  扫过（后者让 trajopt 报假碰撞）。
- **拉的调参**：深而居中的握 → 几乎不打滑 → EE 行程≈抽屉行程，于是把 `PULL_TRAVEL` 收到 0.17、
  恰好开到底就停，少扫 plate（扰动 ~32 mm → ~5–18 mm）。

## 7. **多轮交互 harness**（cap-x 之前没有的"边开边看、撞了就 reset"闭环）
- `interactive.py` + `ictl.sh`：**一次加载 env + planner（~2 min 编译只付一次）**，之后用文件
  收件箱在**持久命名空间**里逐段执行 Python、段间看 sensing+GT、`reset_env(s)` 重置到初态/某 seed。
  上面 1–6 几乎所有发现都是靠它快速试出来的。这正是 self-evolve 交互式评估器需要的形状。
- 配套 `diagnose_corridors.py`（无 planner 的走廊净空快诊）。

## 8. 基础设施
- `scripts/start_capx_services.sh`：一条命令拉起所有默认 service（除 openrouter）；molmo 用专用
  `.venv-molmo`（vLLM）+ `--enforce-eager`。AGENT.md 把"开工先起服务"写进流程。
- session 1 的补丁：task 指令/评分目标对齐（BDDL）、curobo 死路修复（`get_ee_pose` + warp 兼容）。

---

## 一句话
**真正把成功率从 3/5 顶到确定性 5/5 的，是把"最后 5 cm 的精抓"从随机 trajopt 换成
确定性 `solve_ik`+走廊线性插值；其余（深度碰撞世界、sensing 安全门、descend/retreat 的
RRT-兜底、几何感知抓取、交互 harness）是让这一招能稳定落地、且全程不撞物的支撑。**
最大的 cap-x 缺口：**没有一个短程确定性落座原语**，以及 **RRT 在近结构处静默失败/卡死**。
