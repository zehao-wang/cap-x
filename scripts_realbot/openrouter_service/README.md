# OpenRouter 代理服务 / OpenRouter proxy

真机 PIPER 的 LLM（以及 VDM differencing model）走 **OpenRouter Gemini**，经本地
代理在 `:8110` 暴露一个 OpenAI 兼容的 `/chat/completions` —— **不是** robosuite/libero
用的 Qwen vLLM。实现在 `capx/serving/openrouter_server.py`，`run_agent0_piper_interactive.sh`
会自动起它；本目录是同一启动方式的「单一真相」，方便你单独手动拉起。

## 依赖

- cap-x 主环境 `.venv/bin/python`（含 `openai` / `fastapi` / `tyro` / `uvicorn`）
- 仓库根的 `.openrouterkey`（git-ignored）：`echo 'sk-or-v1-...' > .openrouterkey`

## 启动

```bash
# 默认 .openrouterkey + 端口 8110
./run_openrouter_service.sh

# 自定义（参数透传给 openrouter_server.py）
./run_openrouter_service.sh --key-file .openrouterkey --port 8110

# 用环境变量覆盖默认
OPENROUTER_PORT=8110 OPENROUTER_KEY_FILE=.openrouterkey ./run_openrouter_service.sh
```

裸跑时回落到 `OPENROUTER_KEY_FILE`（默认 `.openrouterkey`）+ `OPENROUTER_PORT`
（默认 8110）；只要带了任何参数就完全照传，不再注入默认。

## 联调

`run_agent0_piper_interactive.sh` 把 LLM 调用路由到 `http://localhost:8110/chat/completions`
但**不会**重复起已在跑的实例（端口 UP 就跳过）。手动起好后再跑 launcher 即可复用。
