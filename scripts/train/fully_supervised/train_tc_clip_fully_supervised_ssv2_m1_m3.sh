# fully-supervised SSV2 training with M1 Motion-only-8 + M3 TemporalDeltaHead
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}

protocol=fully_supervised
dataset_name=ssv2
data=${protocol}_${dataset_name}

expr_name=tc_clip_ssv2_mo8_tdh
trainer=tc_clip
use_wandb=true

torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${protocol} \
data=${data} \
output=workspace/expr/${data}/${expr_name}/${data}_${expr_name}_${trainer} \
trainer=${trainer} \
use_wandb=${use_wandb} \
sample_type=motion \
motion_sampler.mode=motion \
num_frames=8 \
temporal_delta.enable=true \
temporal_delta.mode=adjacent \
temporal_delta.alpha=0.5 \
temporal_delta.hidden_dim=256 \
temporal_delta.dropout=0.2 \
temporal_delta.detach_visual=true
