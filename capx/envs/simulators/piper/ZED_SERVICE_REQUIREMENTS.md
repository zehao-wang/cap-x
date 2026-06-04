# ZED 相机服务 —— 实现规范 / ZED Camera Service Spec

一个常驻服务，**独占** ZED 2i 相机，计算**度量深度**（例如基于左/右目的 TRI-Stereo），并通过
**Unix domain socket** 对外返回**对齐好的 RGB-D**。下面是必须满足的协议、数据契约与就绪要求。

---

## 1. 传输 / Transport

- **Unix domain socket**（`AF_UNIX`, `SOCK_STREAM`），同机最快、无 TCP 栈开销。
- 像素作为 **raw bytes** 跟在 JSON header 之后传（**不要** JSON/base64 编码像素）。
- socket 路径默认 `/tmp/piper/zed.sock`（可配置）。
- 1080p 下每帧 ≈ 14 MB（RGB 6.2 MB + depth float32 8.3 MB）。若将来要高帧率连续流，
  可加 POSIX 共享内存快路径（socket 只传 header + shm 块名）；当前不需要。

---

## 2. 线格式 v1 / Wire format

帧 = `4 字节大端无符号长度 L` + `L 字节 payload`。

**请求**（payload = UTF-8 JSON）：
```json
{"v": 1, "method": "get_frame"}
{"v": 1, "method": "ping"}
{"v": 1, "method": "set_camera", "auto": true}
{"v": 1, "method": "set_camera", "auto": false, "exposure": 50, "gain": 50}
```

**`get_frame` 响应** payload：
```
[4 字节大端 H = header 长度][H 字节 UTF-8 JSON header][rgb raw bytes][depth raw bytes]
```
header：
```json
{
  "v": 1, "ok": true,
  "width": 1920, "height": 1080, "timestamp_ns": 1717,
  "rgb":   {"dtype": "uint8",   "shape": [1080,1920,3], "order": "RGB", "nbytes": 6220800},
  "depth": {"dtype": "float32", "shape": [1080,1920],   "unit": "m", "invalid": "nan", "nbytes": 8294400},
  "intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]],
  "baseline_m": 0.12
}
```
其后紧跟 `rgb.nbytes` 字节（行优先 RGB）、再 `depth.nbytes` 字节（行优先 float32）。

**`set_camera` 响应** = 仅 JSON（无尾随像素）。运行时调曝光/增益：`auto:true` 恢复 ZED 自动
AEC/AGC；否则按给定的 `exposure`/`gain`（0..100）设固定值（设值即关掉 auto）。服务用与 `get_frame`
同一把相机锁串行化，不会与抓帧竞争。
```json
{"v":1,"ok":true}
{"v":1,"ok":false,"error":"camera not open"}
```

**`ping` / 错误响应** = 仅 JSON（无尾随像素）：
```json
{"v":1,"ok":true,"camera_open":true,"width":1920,"height":1080,"fps":30}
{"v":1,"ok":false,"error":"camera not ready"}
```

---

## 3. 数据契约 / Data contract（请钉死）

1. **rgb**：uint8 `(H,W,3)`，**RGB 通道序**（不是 BGR）。
2. **depth**：float32 `(H,W)`，**单位米**，**无效 = NaN**；与 rgb **空间对齐**且 **同 H×W**。
3. **rgb 与 depth 来自同一次 grab**（同一帧曝光）。
4. **intrinsics**：3×3 `K`，对应**返回分辨率**（若内部缩放，返回缩放后的 K）。只需内参，**不需要外参/位姿**。

---

## 4. 就绪自检 / Readiness（重要）

- 启动时打开相机（ZED USB3 链路训练常需多次 `open()`，建议每次超时 ~15s、总预算 ~180s 退避重试）。
- 相机 open 后**先自己跑一次 `get_frame` 自检**（grab + depth 成功、形状/类型符合 §2/§3）。
- **自检通过才对外可连 / 才算「开启成功」**；就绪前不要开放 socket（或 `get_frame` 必须回 `ok:false`）。
  → 保证：**只要 socket 可连，请求方就一定能拿到合法数据。**

---

## 5. 生命周期 / 配置 / Lifecycle

- **干净退出**：SIGTERM/SIGINT 时关相机、删 socket 文件（及 shm，若用）。
- **可配置项**：分辨率（默认 1920×1080）、fps（默认 30）、曝光/增益、深度 backend+variant、
  权重目录、socket 路径、open 超时/总预算。
