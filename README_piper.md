# cap-x on Agilex PIPER — 使用手册

本仓库已经集成好 Agilex PIPER 6-DOF 单臂 + ZED 2i 相机的实机 setup，
LLM agent 通过代码执行方式驱动机械臂做 pick-and-lift 一类任务。

这个 README 是**给你自己用的顺手清单**：从开机到 agent 跑起来的完整路径，
以及每一环出问题时怎么自查。

---

## 总览

```
硬件链路
  candleLight USB-CAN ──► can0/can1 ──► PIPER 控制盒 ──► 6-DOF 臂
  ZED 2i ──► USB ──► zed_bridge.py 子进程 ──► _Zed2iBridge (RGB + depth)
  Wrist RealSense D435* ──► USB ──► pyrealsense2 ──► obs["robot0_eye_in_hand"]

软件链路
  env_configs/real/piper_real.yaml
    └─ PiperRealLowLevel          (CAN + ZED + wrist RealSense 融合成 obs/step)
    └─ PiperControlApi            (暴露给 LLM 的函数)
         ├─ sample_grasp_candidates / choose_grasp_candidate
         ├─ sample_grasp_pose     (可传 grasp_index；支持用户挑候选)
         ├─ goto_pose             (pyroki IK + 多中间轨迹点平滑执行)
         ├─ open_gripper / close_gripper
    └─ PiperRealPickCodeEnv       (多轮 code-exec agent)
  api_servers:
    └─ SAM3 @ 8114 (拉起)
  LLM: OpenRouter (.openrouterkey) → default Gemini 3.1 Pro (preview)
```

核心配置文件：**`env_configs/real/piper_real.yaml`**。改 task 改 prompt 都在这里。

---

## 第一次上手：一次性设置

### 1. 依赖

```bash
# base 环境
uv sync

# Piper SDK + CAN
uv pip install piper-sdk python-can pyrealsense2 pyyaml

# ZED：按 robodata_Agilex README 建一个独立 conda env（Python 3.10 + pyzed + numpy<2）
conda create -n zed_bridge python=3.10 -y
conda activate zed_bridge
# ...按 ZED SDK 说明装 pyzed
pip install "numpy<2"
```

### 2. OpenRouter API key

在仓库根目录放一个 `.openrouterkey` 文件（一行一个 key）：

```bash
echo "sk-or-v1-xxxxxxxx" > .openrouterkey
```

> `.openrouterkey` 已被 git 忽略，不会进版本库。

### 2.5. 起 OpenRouter 本地代理

cap-x 不直接打 openrouter.ai —— LLM 调用走本地 `capx/serving/openrouter_server.py`
代理（默认端口 8110），由它读 `.openrouterkey` 转发出去。**跑 agent 前必须先把这个代理拉起来**，
否则 launch 会立刻报连不上 `http://localhost:8110/chat/completions`。

最简单：另外开一个 terminal 前台跑（看日志方便）：

```bash
uv run --no-sync --active capx/serving/openrouter_server.py \
    --key-file .openrouterkey --port 8110
```

后台跑（推荐每次开机就拉起来）：

```bash
mkdir -p logs
nohup uv run --no-sync --active capx/serving/openrouter_server.py \
    --key-file .openrouterkey --port 8110 > logs/openrouter_server.log 2>&1 &
```

确认存活：

```bash
curl -s http://localhost:8110/healthz   # 或者直接打 8110 看是否 connection refused
```

### 3. 环境变量（塞进 `~/.bashrc`）

```bash
# ZED bridge
export ZED_BRIDGE_PYTHON=$(conda run -n zed_bridge which python)
export PIPER_ZED_BRIDGE=$HOME/Documents/Projects/robodata_Agilex/camera/zed_bridge.py

# PIPER URDF（IK 和 viser 都要用）
export PIPER_URDF_PATH=$HOME/Documents/Projects/robodata_Agilex/assets/piper_description/urdf/piper_description.urdf

# 相机外参
export PIPER_CAMERA_EXTRINSICS=$HOME/Documents/Projects/cap-x/env_configs/real/piper_zed_extrinsics.yaml

# Wrist RealSense（只有多台 RealSense 时需要 serial；position/rpy 是 wrist link -> color optical frame）
export PIPER_WRIST_CAMERA_SERIAL=
export PIPER_WRIST_CAMERA_POSITION="0 0 0"
export PIPER_WRIST_CAMERA_RPY_RADIANS="0 0 0"

# CAN（改成你实际用的那根）
export PIPER_CAN_CHANNEL=can1
export PIPER_CAN_INTERFACE=socketcan
export PIPER_CAN_BITRATE=1000000
```

---

## 每次开机流程

### 0. 起 OpenRouter 本地代理（如果还没在跑）

```bash
pgrep -af openrouter_server || \
  nohup uv run --no-sync --active capx/serving/openrouter_server.py \
      --key-file .openrouterkey --port 8110 > logs/openrouter_server.log 2>&1 &
```

详细见 [上面的 2.5 节](#25-起-openrouter-本地代理)。

### 1. 起 CAN

```bash
sudo bash scripts/setup_can.sh
# 输出列出的接口里应该有你的 canN 且是 UP
```

### 2. 确认哪根 canN 是你要的臂

```bash
uv run --no-sync --active scripts/piper_identify_arms.py
# 手动晃一下要用的那条臂，看哪个 canN 的数值在变
```

如果你的臂不是 `can1`，`export PIPER_CAN_CHANNEL=canX`。

### 3. 确认臂在 slave 模式

cap-x 驱动的臂必须是 **slave 模式**（master 模式不响应 JointCtrl）：

```bash
uv run --no-sync --active scripts/piper_set_mode.py --channel can1 --mode slave
# 如果是从 master 切到 slave，需要给机械臂断电重启一次
```

### 4. 一次外参 sanity check（推荐）

```bash
uv run --no-sync --active scripts/cam_calibration/piper_visualize_zed_extrinsics.py
```

打开 http://localhost:8201，确认：
- URDF 跟真机关节同步
- 绿色 frustum 跟 ZED 物理位置吻合
- frustum 画面实时刷新，画面内容 ≈ ZED 指向的方向

外参错了后面全是空谈。详细验证方法见
[`scripts/cam_calibration/piper_zed_calibration.md`](scripts/cam_calibration/piper_zed_calibration.md)。

---

## 标定 ZED 外参

**每次物理挪动 ZED 都要重标。**

```bash
uv run --no-sync --active scripts/cam_calibration/piper_calibrate_zed_extrinsics.py
```

完整步骤、残差判定、避坑指南：**[`scripts/cam_calibration/piper_zed_calibration.md`](scripts/cam_calibration/piper_zed_calibration.md)**。

关键点速记：
- 标定板**刚性固定在夹爪上**（关键）
- 至少 12 组姿态，每组都要**倾斜板子**（不要只平移）
- 残差 `< 10 mm` 优秀；`10–20 mm` 能用；`> 20 mm` 重采

---

## 跑 agent

### 1. 改任务 prompt（可选）

编辑 `env_configs/real/piper_real.yaml`：

```yaml
env:
  cfg:
    task_only_prompt: "pick up the red cube and lift it"   # ← 改这里
    prompt: |
      ...                                                   # ← 或这里
```

- `task_only_prompt`：给 agent 的**单句目标**
- `prompt`：完整 system prompt，包含 API 用法示例。如果你想让 agent 自己推理调用顺序，把 `Typical sequence:` 那段删掉

### 2. 启动

```bash
PIPER_ZED_DEPTH_MODE=NEURAL uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/real/piper_real.yaml
```

会发生什么：
- 自动拉起 SAM3 server（8114）
- 连 CAN、ZED 和 wrist RealSense（`obs["robot0_eye_in_hand"]`）
- 启动时自动回 Home（6 轴全 0 的 rest pose）
- 打开 web UI（默认 http://localhost:8200）看 agent 每一轮 IO
- Agent 多轮交互：代码 → 执行 → 看 stdout/obs → 下一段代码 → ... → `FINISH`
- 视频录到 `outputs/piper_real/`

**第一次跑**：**一只手放在急停上**。外参残差 + IK 误差 + 分割误差叠起来，
撞桌/夹自己的概率不低。

### 3. 演示任务识别（10s 视频 → 任务摘要）

当 web UI 在等待用户输入时，可以输入：

```text
/demo_task
```

系统会自动：
- 从当前相机流采 10 秒视频，按 1fps 抽帧（共约 10 帧）
- 调用当前视觉模型做一句话任务识别
- 回来询问你是否执行该任务：
  - 输入 `yes`：采用识别结果作为任务
  - 输入 `no`：不采用识别结果
  - 输入任意新文本：用你输入的新任务覆盖

### 4. 暴露给 LLM 的 API

```python
# 都已经 import 到执行环境里
cands = sample_grasp_candidates("red cube", top_k=5)   # 返回候选列表（按列表索引选）
pos, quat = choose_grasp_candidate(0)                  # 用户输入 index 后选择
# 也可一步到位：
# pos, quat = sample_grasp_pose("red cube", grasp_index=0)
goto_pose(pos, quat, z_approach=0.1)                   # 平滑多轨迹点执行
open_gripper() / close_gripper()
# numpy 要自己 import
```

固定 planner 参数在 `env_configs/real/piper_real.yaml`:
- `piper_planner_timesteps`: `goto_pose` 每次调用固定轨迹点数（总步数）
- `piper_min_target_z`: 当输入目标 `z < piper_min_target_z` 时钳制到的最小 z（默认 0.01m）

想加新 API：在 `capx/integrations/piper/control.py` 的 `PiperControlApi.functions`
字典里新增，然后在 prompt 里提一下。

---

## 分步调试顺序

从风险低到高，逐步验证整条链：

| 步 | 命令 | 验证什么 |
|---|---|---|
| 1 | `scripts/cam_calibration/piper_visualize_zed_extrinsics.py` | URDF/CAN、ZED、外参 |
| 2 | 手写 5 行调 `PiperControlApi.get_object_pose("red cube")` | SAM3 感知 |
| 3 | 手写调 `sample_grasp_pose` + `goto_pose(..., z_approach=0.15)` 不夹 | IK + 执行 + 外参一致性 |
| 4 | 手写完整 pick-and-lift 序列 | 整条无 agent 流程 |
| 5 | `capx/envs/launch.py` 启动 agent | LLM 控制 |

前 4 步都通了再上 agent，省很多时间。

---

## 常见问题

### 硬件 / CAN

| 症状 | 原因 | 修法 |
|---|---|---|
| `setup_can.sh` 报 "no candleLight adapter" | USB 没插 / 没识别 | `lsusb` 看 `1d50:606f`；换口 |
| `ConnectPort()` 卡住 | canN 没 UP / 另一个进程占用 | `ip -brief link show`；`sudo ip link set can1 down && up` |
| 发命令臂不动 | 在 master 模式 | `scripts/piper_set_mode.py --mode slave`，断电重启 |
| 关节读数对但 goto_pose 不动 | 同上 | 同上 |

### ZED

| 症状 | 原因 | 修法 |
|---|---|---|
| `_Zed2iBridge` 启动失败 | `ZED_BRIDGE_PYTHON` 指错了 | `conda run -n zed_bridge which python` 验证 |
| 一直 `rgb is None` | bridge 起来了但没出帧 | 另外开 terminal 直接跑 `$ZED_BRIDGE_PYTHON $PIPER_ZED_BRIDGE` 看错误 |

### 外参 / IK

| 症状 | 原因 | 修法 |
|---|---|---|
| agent 抓取偏几 cm | 外参残差 > 10 mm | 重标；或者用更大物体 |
| `from_matrix` det<0 报错 | 板子没固定 / 旋转多样性差 | 参见 `scripts/cam_calibration/piper_zed_calibration.md` |
| viser frustum 位置完全错 | `rpy_radians` 约定搞反了 | 必须是 extrinsic 小写 `"xyz"` |
| goto_pose 报 IK 失败 | 目标超出工作空间 | 打印 `pos` 查合理性；`z_approach` 加大 |

### LLM

| 症状 | 原因 | 修法 |
|---|---|---|
| launch 立刻退出说 API key | `.openrouterkey` 没放 | 放到仓库根目录 |
| launch 报 `Connection refused` / 打不到 `localhost:8110` | OpenRouter 本地代理没起来 | `pgrep -af openrouter_server` 看一下；按 [2.5 节](#25-起-openrouter-本地代理) 拉起来 |
| 代理日志里 `401 / invalid api key` | `.openrouterkey` 内容错或过期 | 去 openrouter.ai 重新生成、覆盖 `.openrouterkey`，重启代理 |
| agent 永远不 `FINISH` | prompt 里没讲清 "FINISH" | 看 `multi_turn_prompt`，确保指令明确 |

---

## 相关文件

- `env_configs/real/piper_real.yaml` — 主配置
- `env_configs/real/piper_zed_extrinsics.yaml` — ZED 外参（标定产出）
- `capx/envs/simulators/piper_real.py` — Low-level env entrypoint, same public style as `franka_real.py`
- `capx/envs/simulators/piper/` — setup, motion, IO, Viser, camera bridges, shared helpers
- `capx/integrations/piper/control.py` — `PiperControlApi`（agent 调用的函数）
- `capx/envs/tasks/piper/piper_pick.py` — Code-exec task env
- `capx/envs/launch.py` — agent 启动入口
- `scripts/cam_calibration/piper_calibrate_zed_extrinsics.py` — 外参标定
- `scripts/cam_calibration/piper_visualize_zed_extrinsics.py` — 外参可视化
- `scripts/cam_calibration/piper_zed_calibration.md` — 标定详细流程
- `scripts/piper_identify_arms.py` — canN ↔ 物理臂映射
- `scripts/piper_set_mode.py` — master/slave 模式切换
- `scripts/setup_can.sh` — CAN 初始化
