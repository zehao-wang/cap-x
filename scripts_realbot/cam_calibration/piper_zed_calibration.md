# PIPER ↔ ZED 2i 外参标定流程

目的：标定 ZED 2i 相机在 Agilex PIPER 机械臂 base 坐标系下的位姿
（`T_base_cam`），保存到 `env_configs/real/piper_zed_extrinsics.yaml`，
供 `PiperRealLowLevel` 把 ZED 观测转换到 base 系使用。

这是 **eye-to-hand** 标定：相机固定在外面不动，标定板贴在夹爪上跟着手臂动。

---

## 1. 硬件准备

- **标定板**：用 `scripts_realbot/cam_calibration/generate_calibration_board.py`
  生成默认的 **5×6 格 20 mm 方格棋盘**（板尺寸 100×120 mm，内角 4×5），
  打印后贴在硬板（亚克力 / 泡沫板 / 硬纸板）上。其它尺寸用
  `--cols/--rows/--square` 覆盖。
- **把标定板刚性固定到夹爪上**。用胶带/夹子/螺丝固定到一个夹爪平面，
  确保在整个标定过程中板子**相对夹爪绝对不动**（哪怕 1–2 mm 的滑动都会
  严重拉高残差）。夹爪张合不影响标定板即可。
- **ZED 2i** 固定安装到最终工作位置——标完后别再动相机，否则外参失效。
- CAN 总线（默认 `can1` @ 1 Mbps）和 ZED bridge 环境变量就位：
  - `PIPER_ZED_BRIDGE` → `zed_bridge.py` 路径
  - `ZED_BRIDGE_PYTHON` → 能 import `pyzed` 的 Python 解释器

## 2. 采集 + 求解

```bash
uv run --no-sync --active scripts_realbot/cam_calibration/piper_calibrate_zed_extrinsics.py
```

交互流程（默认 12 组）：

1. 把手臂移到一个姿态，确保标定板在 ZED 画面里**完整可见**、**光照均匀**。
2. 在终端按 **ENTER** 采集一组。
   - `s` 跳过，`q` 提前结束
3. 重复 12 次。

采集要点：

- **平移要分散**：不要把 12 组全堆在同一区域。工作空间的 x/y/z 都覆盖。
- **旋转更关键**：每次都要**倾斜/翻滚标定板**，不要只平移。至少让板子的法线
  方向绕三个轴都有 ≥ 30° 的变化。**旋转多样性不足是标定翻车的头号原因。**
- 板子不能出画、不能被夹爪遮挡、不能过曝/过暗。

采集期间 viser（默认 http://localhost:8201）会实时显示：
- ZED 预览 + 检测到标定板时叠加的 XYZ 轴
- PIPER URDF 跟随真机关节
- 每组采集后新增一个小坐标系 `/captures/pose_XX`

采集完成后脚本会：
1. 跑 `cv2.calibrateHandEye`（eye-to-hand 反向输入，`CALIB_HAND_EYE_PARK` 方法）
2. 打印 **board-on-gripper 平移残差 std**（三个轴分别）
   - `< 10 mm`：优秀
   - `10–20 mm`：能用
   - `> 20 mm`：**别信**，重采（增加旋转多样性、确认板子固定）
3. 写 `env_configs/real/piper_zed_extrinsics.yaml`：
   ```yaml
   position:    [x, y, z]            # 米
   rpy_radians: [roll, pitch, yaw]   # 外参 XYZ extrinsic，弧度
   ```
4. 在 viser 里画**绿色相机 frustum**（位于解出的 `T_base_cam`），
   frustum 的图像平面显示最近一帧 ZED 画面。按 ENTER 退出。

## 3. 独立可视化验证

标完后想再验证 / 后续随时检查，不用重跑标定：

```bash
uv run --no-sync --active scripts_realbot/cam_calibration/piper_visualize_zed_extrinsics.py
```

可视化内容（http://localhost:8201）：

- `/base` — PIPER base 坐标系
- PIPER URDF — 关节跟随真机实时更新
- `/calibrated_zed` — 标定出来的 ZED 位置，绿色 frustum，图像平面播放实时 RGB

可选参数：

- `--extrinsics <path>` — 指定别的 YAML
- `--no-piper` — 不连 CAN（只看 frustum + 静态 URDF）
- `--no-zed` — 不起 ZED bridge（只看 frustum 位置，没图像）
- `--viser-port <port>` — 换端口

**三项肉眼检查**（按严格度递增）：

1. **位置**：绿色 frustum 应该跟真实 ZED 的物理位置大致吻合
   （几厘米以内）。
2. **朝向**：frustum 指向应对准工作区；frustum 图像里看到的场景应该就是
   frustum 指向的方向。
3. **重合度**（最硬核）：把手臂移到 ZED 画面里能看到的位置。
   viser 里 URDF 的夹爪 vs frustum 图像里真实夹爪，沿相机射线方向应重合。
   这一步能同时验证 position、rotation、关节状态三者是否一致。

## 4. 常见坑

| 症状 | 可能原因 | 修法 |
|---|---|---|
| `Non-positive determinant` 报错 | 旋转解垃圾（欠定 / 板子松动） | 检查板子固定；重采并增加旋转多样性 |
| 残差 > 50 mm | 板子松动 / 12 组都几乎同姿态 | 重贴板子；每次刻意倾斜不同轴 |
| 残差 < 20 mm 但 frustum 位置明显偏 | base / cam 约定方向理解错 | 跟 `piper_real.py` 里 `_load_extrinsics` 对一下约定（`rpy_radians` 是 extrinsic `"xyz"`） |
| `board not detected` | 棋盘被遮挡/过曝/不完整 | 调光、把板子整个塞进画面 |
| viser 里 URDF 不动 | CAN 没连上 / 手臂没上电 | `scripts_realbot/setup_can.sh`；检查 `can1` `ip link` |

## 5. 约定备忘

- `rpy_radians` 是 **extrinsic `"xyz"`**（小写，固定坐标系绕 x→y→z 依次旋转）。
  `piper_real.py:_load_extrinsics` 读 YAML 用的就是这个约定，标定脚本写
  YAML 也用这个约定。**不要改成大写 `"XYZ"`（intrinsic）**，两者给出的
  旋转矩阵不同。
- `position` 是**相机原点**在 base 系下的坐标（米）。
- Piper SDK `GetArmEndPoseMsgs().end_pose` 的 `RX/RY/RZ_axis` 也按 extrinsic
  `"xyz"`（弧度 = 原值 × 1e-3 × π/180）解读，见
  `piper_calibrate_zed_extrinsics.py:read_piper_pose`。
