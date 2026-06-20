# fully-supervised SSV2 evaluation for the original TC-CLIP baseline
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}

protocol=fully_supervised
dataset_name=ssv2
data=${protocol}_${dataset_name}
expr_name=tc_clip_ssv2_baseline_uniform16
resume=${RESUME:-workspace/expr/${data}/${expr_name}/${data}_${expr_name}_tc_clip/best.pth}
trainer=tc_clip

torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${protocol} \
data=${data} \
eval=test \
output=workspace/results/${data}/${expr_name}_${trainer} \
resume=${resume} \
trainer=${trainer} \
sample_type=uniform \
num_frames=16 \
temporal_delta.enable=false
