#!/bin/bash
network=$1
gpu_number=$2
output_dir=${network}
output_dir="cifar10_${output_dir}_base_network"


export PYTHONPATH="$(pwd)"

CUDA_VISIBLE_DEVICES="${gpu_number}" python src/cifar10_channel_prune/train.py \
  --output_dir=${output_dir} \
  --data_format="NCHW" \
  --network=${network} \
