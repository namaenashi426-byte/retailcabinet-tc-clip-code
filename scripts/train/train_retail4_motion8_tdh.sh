#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/path/to/RetailCabinet-4/videos}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

torchrun --nproc_per_node="${GPUS_PER_NODE}" main.py -cn fully_supervised \
  data=fully_supervised_cjj \
  output=workspace/expr/retail4_motion8_tdh \
  trainer=tc_clip \
  use_wandb=false \
  retail4.root="${DATA_ROOT}" \
  sample_type=motion \
  motion_sampler.mode=motion \
  num_frames=8 \
  temporal_delta.enable=true \
  temporal_delta.mode=adjacent \
  temporal_delta.alpha=0.5 \
  temporal_delta.hidden_dim=256 \
  temporal_delta.dropout=0.2 \
  temporal_delta.detach_visual=true
