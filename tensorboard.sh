#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
read -r LOG_DIR PORT < <(
  /home/amax/HDD1/gl_user/GL_env/vllm/bin/python -c \
    'from config_train import TrainConfig; c = TrainConfig(); print(c.tensorboard_log_dir, c.tensorboard_port)'
)
exec /home/amax/HDD1/gl_user/GL_env/vllm/bin/tensorboard \
  --logdir "$LOG_DIR" --host 127.0.0.1 --port "$PORT"
