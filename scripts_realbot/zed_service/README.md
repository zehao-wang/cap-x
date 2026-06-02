# ZED 2i 深度服务 / ZED depth service

按 `capx/envs/simulators/piper/ZED_SERVICE_REQUIREMENTS.md`（线格式 v1）实现的常驻服务：
**独占** ZED 2i，用 **TRI-Stereo**（学习深度，非 SDK）算度量深度，经 **Unix domain socket**
返回对齐好的 RGB-D。cap-x 侧瘦客户端是 `capx/envs/simulators/piper/zed_service_client.py`。

## 依赖

深度模型与 pyzed 都在 raiden 的 venv 里（cap-x 主环境不持有任何深度依赖）：

- `pyzed.sl`（ZED SDK，numpy<2）
- `raiden.depth.tri_stereo`（TRI-Stereo predictor，TensorRT > ONNX 自动选）
- 权重：`raiden/weights/tri_stereo/stereo_{c32,c64}.{onnx,engine}`

## 启动

```bash
# 默认 1080p@30、socket /tmp/piper/zed.sock、TRI-Stereo c64（TRT 优先）
./run_zed_service.sh

# 自定义
./run_zed_service.sh --socket /tmp/piper/zed.sock --width 1920 --height 1080 \
                     --fps 30 --variant c64
```

`run_zed_service.sh` 用 `RAIDEN_VENV`（默认指向本机 raiden venv）里的 python 跑
`zed_depth_service.py`，所有参数透传。

## 行为契约（见 REQUIREMENTS）

- **就绪自检**：相机 open（最多 ~180s 退避重试）后先自己跑一次 `get_frame` 自检；
  **自检通过才创建 socket**。⇒ socket 可连即保证能拿到合法 RGB-D。
- **数据**：rgb uint8 `(H,W,3)` RGB 序；depth float32 `(H,W)` 米、无效=NaN，与 rgb 同帧同尺寸。
- **干净退出**：SIGTERM/SIGINT 关相机、删 socket 文件。
- **配置**：`--width/--height/--fps/--serial/--exposure/--gain/--variant/--weights-dir/`
  `--socket/--open-timeout-sec/--open-deadline-sec`。

## 联调

服务起来后，把 cap-x 的 `piper_zed_source` 切到 `service`（默认 `bridge`），
`piper_zed_service_socket` 指到同一 socket 路径。
