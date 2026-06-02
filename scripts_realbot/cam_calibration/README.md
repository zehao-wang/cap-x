# PIPER ↔ ZED 2i 外参标定

工具集：把 ZED 2i 的位姿标定到 Agilex PIPER 机械臂的 base 坐标系下，结果写到
`env_configs/real/piper_zed_extrinsics.yaml`，供 `PiperRealLowLevel` 把相机
观测转到 base 系使用。

这是 **eye-to-hand** 标定：相机外置不动，标定板贴在夹爪上随手臂动。

---

## 文件清单

| 文件 | 用途 |
|---|---|
| [`generate_calibration_board.py`](generate_calibration_board.py) | 生成可打印的棋盘 / ChArUco PDF（默认 100×120 mm，A4 排版） |
| [`piper_calibrate_zed_extrinsics_auto.py`](piper_calibrate_zed_extrinsics_auto.py) | **手动按钮**标定：在 viser 里按按钮，自己拖着机械臂走点（推荐） |
| [`piper_calibrate_zed_extrinsics.py`](piper_calibrate_zed_extrinsics.py) | **终端交互**标定：手动摆姿态、按 ENTER 采集（早期版本，仍可用） |
| [`piper_visualize_zed_extrinsics.py`](piper_visualize_zed_extrinsics.py) | 标完后独立可视化验证（URDF + ZED frustum + 实时画面） |
| [`piper_zed_calibration.md`](piper_zed_calibration.md) | 标定流程详解、残差判定、避坑指南 |

---

## 推荐流程

### 0. 一次性：打印标定板

```bash
uv run --no-sync --active scripts_realbot/cam_calibration/generate_calibration_board.py
```

产物在 `data/calib_boards/` 下（PNG + PDF）。打印步骤：

1. 用 PDF 文件打印，**100% / "Actual size"**，禁用 "fit to page"。
2. 拿真尺子量页面底部那条 `50 mm scale` 刻度尺，确认正好 50 mm 才放心。
3. 沿四角十字标记裁下来，贴到刚性背板（亚克力 / 泡沫板 / 硬卡纸）。

可选参数：

```bash
# ChArUco 板（更鲁棒，对光照/部分遮挡更友好）
... generate_calibration_board.py --board charuco

# 自定义尺寸：6×8 个 15 mm 方格 = 90×120 mm
... generate_calibration_board.py --squares-x 6 --squares-y 8 --square 15
```

页面下方那行 `calib flags:` 直接抄到标定脚本的命令行里：

```text
calib flags:  --cols 4  --rows 5  --square 0.0200
```

### 1. 手动按钮标定（推荐）

```bash
uv run --no-sync --active scripts_realbot/cam_calibration/piper_calibrate_zed_extrinsics_auto.py
```

脚本不会自己动机械臂——你自己用手拖着走点位，按按钮采集。

**界面**（http://localhost:8300）：
- 左侧 ZED 实时画面，板检测到时叠 XYZ 三轴
- URDF 跟随真机关节
- GUI 面板按钮：

  | 按钮 | 行为 |
  |---|---|
  | ✋ Open gripper | 张开夹爪 |
  | ✊ Close gripper on board | 闭合夹爪，监测 effort，碰到板就 latch（不会越夹越紧） |
  | 🔓 Disable arm torque | 关掉电机扭矩，机械臂可徒手拖动（**点之前先扶住臂！** 否则会因重力下垂） |
  | 🔒 Re-enable arm torque | 重新上电锁定 |
  | 📸 Capture this pose | 读 SDK 当前 end pose + 检测板 → 存一组 |
  | 🗑️ Discard last capture | 撤销最后一次 |
  | 🧮 Solve & save calibration | ≥ 3 组就能解，求解后写 YAML 并画绿色 frustum |

**典型流程**：
1. ✋ 张开 → 把板塞进夹爪 → ✊ 闭合（latch 在板的厚度上）。
2. 🔓 关扭矩（**先扶住臂**）。
3. 拖到板在 ZED 里能完整看到的位姿 → 📸 采集。每次**倾斜板子**让旋转覆盖
   多个轴 → 重复 8–12 次。旋转多样性比平移多样性重要。
4. 🧮 求解 → YAML 写入 `env_configs/real/piper_zed_extrinsics.yaml`，
   viser 加上绿色 frustum。残差 std > 20mm 会有警告，可以继续多采几组
   再次点 Solve 复算。
5. 🔒 重新锁定（可选）→ Ctrl+C 退出。

**采集要点**：
- 板在画面里要**完整可见**、光照均匀、不要被夹爪/手挡住
- 每组刻意**倾斜板子**（roll/pitch/yaw 各个方向都试），不要只平移
- 至少 ≥ 8 组，12+ 更稳

常用参数：

```bash
--captures-dir /tmp/calib  # 保存每张采集 RGB
--viser-port 8300          # 可视化端口（默认就是 8300）
--board charuco            # 用 ChArUco 板（默认 checkerboard）
--cols 4 --rows 5 --square 0.020  # 板参数（这就是默认值，等同于
                                  # generate_calibration_board.py 的默认 5×6×20mm 棋盘）
```

### 2. 交互标定（手动摆姿态）

如果自动模式的 12 个预设位姿都不适合（比如相机装的位置很奇葩，或工作空间
受限），可以退回到完全手动版：

```bash
uv run --no-sync --active scripts_realbot/cam_calibration/piper_calibrate_zed_extrinsics.py
```

自己摆好每个姿态，按 ENTER 采集，用法见 [`piper_zed_calibration.md`](piper_zed_calibration.md)。

### 3. 标定后的可视化验证

任何时候想确认外参是否还正确（没人动过相机），不用重标：

```bash
uv run --no-sync --active scripts_realbot/cam_calibration/piper_visualize_zed_extrinsics.py
```

打开 http://localhost:8201 三个肉眼检查：

1. **位置**：绿色 frustum 跟真实 ZED 物理位置吻合（几 cm 内）。
2. **朝向**：frustum 指着工作区；frustum 图像里的画面 ≈ frustum 朝向。
3. **重合度（最硬核）**：把手臂移到 ZED 画面里，URDF 夹爪 vs frustum 图像
   里真实夹爪沿相机射线方向应重合。

---

## 标定输入/输出

**输出 YAML**（`env_configs/real/piper_zed_extrinsics.yaml`）：

```yaml
position:    [x, y, z]            # 米，相机原点在 base 系下的坐标
rpy_radians: [roll, pitch, yaw]   # 外参 XYZ extrinsic（小写），弧度
```

**约定备忘**：

- `rpy_radians` 是 extrinsic `"xyz"`（**小写**，固定坐标系绕 x→y→z 依次旋转）
  ——`piper_real.py:_load_extrinsics` 读 YAML 用的是这个约定。**不要改成大写
  `"XYZ"`（intrinsic），两者给出的旋转矩阵不同。**
- Piper SDK `GetArmEndPoseMsgs().end_pose` 的 `RX/RY/RZ_axis` 也按 extrinsic
  `"xyz"` 解读（弧度 = 原值 × 1e-3 × π/180），见
  [`piper_calibrate_zed_extrinsics.py:read_piper_pose`](piper_calibrate_zed_extrinsics.py)。

---

## 残差判定

求解后脚本会打印 board-on-gripper 平移残差 std（mm）：

| 三轴最大值 | 评估 |
|---|---|
| `< 10 mm` | 优秀 |
| `10–20 mm` | 能用 |
| `> 20 mm` | **别信**——重采（增加旋转多样性、确认板子在夹爪上没滑动） |

更多坑表见 [`piper_zed_calibration.md`](piper_zed_calibration.md) §4。

---

## 什么时候要重标

- ZED 物理位置 / 安装方向 **任何**变动 → 必须重标。
- 机械臂底座移动 → 必须重标。
- 夹爪 / 末端工具更换 → 不影响外参（标的是相机相对 base，跟末端无关），
  但 `TCP_OFFSET` 可能要改。
