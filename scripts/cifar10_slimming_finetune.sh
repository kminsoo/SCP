#!/bin/bash
percent=$1
network=$2
gpu_number=$3

export PYTHONPATH="$(pwd)"

CUDA_VISIBLE_DEVICES="${gpu_number}" python src/cifar10_channel_prune/train_slimming_finetune.py \
  --output_dir=${output_dir} \
  --data_format="NCHW" \
  --network=${network} \
  --percent=${percent} \
