# start language model service
uv run --no-sync --active capx/serving/openrouter_server.py  --key-file .openrouterkey --port 8110

# start camera service
uv run --no-sync --active python -m capx.serving.launch_piper_state_service --config env_configs/real/piper_real.yaml --host 127.0.0.1 --port 8210 --shm-prefix piper_svc

# start the agent
uv run --no-sync --active capx/envs/launch.py --config-path env_configs/real/piper_real_service.yaml