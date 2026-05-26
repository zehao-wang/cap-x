# 可视化 / 回放（viser playback）

> **状态**: ✅ 已实现
> **读 → 写**: 每步观测帧（相机图像 + 位姿 + FOV + joints）→ viser 面板回放 +
> `outputs/trial_NN/viser_history/attempt_NN.npz`
> **实现**: `capx/utils/viser_history.py`（数据/生命周期 facade）、
> `capx/utils/viser_playback_panel.py`（viser GUI/scene）、
> `capx/utils/viser_history_io.py`（`.npz` 持久化）
> **独立性**: 纯调试/可视化功能，**不影响 self-evolve pipeline 的数据流**；与
> [01-interactive-loop.md](01-interactive-loop.md) 的耦合点仅在 reset 时 `new_segment()` /
> 全新 trial 时 `clear()`。

- **每次 reset = 一段新可视化**。所有 segment 常驻内存，界面 **Attempt 下拉**可选看任意一次
  尝试；同时每段按顺序保存到 `outputs/trial_NN/viser_history/attempt_00.npz`、`attempt_01.npz`
  …（与当前 trial 绑定；new trial 即全新开始）。
- **camera frustum 必须符合设定**：3D 中每个相机各自画 frustum，朝向 + 位置取自相机外参，
  视锥张角取相机真实 FOV（由内参 / `cam_fovy` 推出），不再用硬编码默认值。
- **Multi-view 拼接**：一帧里多相机的图像**横向拼成一张** Observation 图显示；3D 里仍每相机
  一个 frustum。Camera 下拉用于选择 Reset View 的目标相机。
- **Reset View = 回到所选相机的当前位置**（当前显示帧里该相机的位姿），不是某个初始视角。

## 模块边界（实现/维护时）

- `ViserFrameHistory`（facade）：只管 frame 数据——segment 分组、`record` / `clear` /
  `new_segment`、溢出二分降采样、把保存委托给 io。simulator 只与它打交道
  （`from capx.utils.viser_history import ViserFrameHistory`）。
- `ViserPlaybackPanel`：拥有一切 viser 相关——Attempt/Timestep/Live/Camera/Reset-View 控件、
  拼接图、frustum、相机父坐标系、orbit 相机吸附；以及"视图状态"（当前看哪段/哪帧、是否
  Live、Reset View 目标相机）。它通过 `segments_getter` 回调读 facade 的 segment。
- `save_frames(frames, path)`：纯 `.npz` 序列化（duck-typed，不 import facade）。
