#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
exec /home/amax/HDD1/gl_user/GL_env/vllm/bin/torchrun \
  --standalone --nproc_per_node=6 train_meteotime.py
