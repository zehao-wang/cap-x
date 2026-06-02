# Short-Term-GOAL —— 本次任务暂存

> 临时 memory：只放「本次正在推进的那个模块」的可执行步骤与进展。
> 模块做完 → 在 `GOAL.md` 对应行标 ✅，然后清空本文件回到这个模板。
> 快超 usage limit 时，先把「这次改了什么 / 还想改什么」归纳到下面。

## 本次模块

Piper 实机交互闭环跑通（不在 §2 pipeline 表内；这是唯一还没 debug/eval 的链路）。
要求：largely 跟随 `run_agent0_qwen36_robosuite_interactive.sh`，但跑本地实机 + OpenRouter
Gemini（**绝不用 qwen3.6**）。

## 可执行步骤

- [x] 1. LLM 选型：确认 launch.py 默认即 `google/gemini-3.1-pro-preview` + `:8110` openrouter 代理；
      qwen3.6 只由 robosuite/libero 脚本 `--model` 覆盖。Piper 脚本显式 pin Gemini。
- [x] 2. 修 web 配置族 bug：`capx/web/server.py::_config_family` 只认 `franka_real` → piper_real
      被误判成 "other"（会被 `CAPX_CONFIG_FAMILIES=real` 过滤掉 + 不受 CAPX_ENABLE_REAL 门控）。
      已加 `or "piper_real" in text`。
- [x] 3. 新建 `scripts_realbot/run_agent0_piper_interactive.sh`：本地实机版，OpenRouter Gemini，
      起 openrouter 代理 + SAM3/GraspNet（web-ui 模式 launch.py 不起 api_servers），设
      CAPX_ENABLE_REAL=1 / CAPX_CONFIG_FAMILIES=real，launch 实机 config。
- [ ] 4. 实机连接 bug：需用户在带显示器的真机上跑脚本、贴报错，再逐轮修（CAN/EnablePiper/ZED bridge）。

## 进展 / 改了什么

- `capx/web/server.py`：`_config_family` 增加 `piper_real` 标记 → 归入 "real" 族。
- 新增 `scripts_realbot/run_agent0_piper_interactive.sh`（已 chmod +x，bash -n 通过）。
- 关键发现：web-ui 模式下 `capx/envs/launch.py:261` 不启动 yaml 的 api_servers，故脚本必须自起
  SAM3/GraspNet；openrouter 代理(:8110) launch.py 也不起，需脚本起。
- 静态扫过 setup.py/motion.py/io.py：未见明显逻辑崩溃；真正的连接 bug 要实机跑出来。

## 还想改什么（收尾归纳）

- 等用户实机跑 `scripts_realbot/run_agent0_piper_interactive.sh` 贴报错 → 逐轮修连接问题（步骤 4）。
- [x] README_piper §总览 已统一为「default Gemini 3.1 Pro (preview)」（用户确认 gemini-3.1-pro-preview 为准）。

## 子任务：ZED 场景相机走「独立服务」，cap-x 只读（仅 Piper 真机）

最终决定（用户拍板，见 `GOAL.md §5`）：ZED 单独起一个服务返回 RGB+depth，**cap-x 只读、不算深度**。
深度模型（TRI-Stereo 等）全在服务侧；服务由 raiden 实现，不在 robodata_Agilex 改东西。

- [x] **回退**早先「cap-x 内跑 TRI-Stereo」的全部改动：删 `piper/tri_stereo.py`、卸 onnxruntime-gpu、
      `git checkout` 还原 zed_bridge.py/setup.py/piper_real.py/base.py/launch_piper_state_service.py/
      pyproject.toml/piper_real.yaml；`robodata_Agilex/camera/zed_bridge.py` 也已还原（repo pristine）。
- [x] **契约文档**移进 cap-x：`capx/envs/simulators/piper/ZED_SERVICE_REQUIREMENTS.md`
      （§2 传输：UDS 控制 + 像素二进制尾随 v1，shm 为未来可选；§3 线格式 v1 钉死；§4 数据契约）。
- [x] **瘦客户端** `capx/envs/simulators/piper/zed_service_client.py::_ZedServiceClient`：UDS、只读、
      drop-in（`start/read_frames->(rgb,depth)/intrinsics_matrix/stop` + 曝光增益 best-effort）。
- [x] **接线（非破坏）**：base.py + piper_real.py + setup.py 加 `piper_zed_source`(service|bridge,默认 bridge)
      + `piper_zed_service_socket`(默认 `/tmp/piper/zed.sock`)；`_Zed2iBridge` 保持原样供标定脚本用。
- [x] **验证**：`tests/test_zed_service_client.py`（fake UDS server round-trip）通过；import/默认值/契约字段已查。
- [x] **服务侧实现** `scripts_realbot/zed_service/`：`zed_depth_service.py`（独占 ZED 2i、DEPTH_MODE.NONE
      取左右目、`raiden.depth.tri_stereo` 算度量深度、UDS 线格式 v1、就绪自检后才 bind socket、
      SIGTERM/SIGINT 干净退出删 socket）+ `run_zed_service.sh`（用 raiden venv 起）+ `README.md`。
  - 验证（无相机）：① loopback —— 真 server 线路 vs 真 cap-x `_ZedServiceClient`，rgb/depth(NaN)/
    内参/ping/干净退出全过；② 真 TRI-Stereo backend 在 raiden venv 实际加载(ONNX c32 CUDA)+ predict
    出 `(H,W)` float32 米深度。py_compile（两 venv）+ `bash -n` 通过。
- [ ] **实机联调**（live-gated，缺真机/显示器）：真起服务开相机跑一次 → 把 `piper_zed_source` 切
      `service`、`piper_zed_service_socket` 指同一路径，端到端跑通 Piper 观测流。
