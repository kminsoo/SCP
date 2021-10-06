#!/bin/bash
logistic_c=$1
sparse_bernoulli=$2
s=$3
gpu_number=$4
network=$5

export PYTHONPATH="$(pwd)"

CUDA_VISIBLE_DEVICES="${gpu_number}" python src/cifar10_channel_prune/train_gumbel_prune.py \
  --data_format="NCHW" \
  --network=${network} \
  --logistic_c=${logistic_c} \
  --sparse_bernoulli=${sparse_bernoulli} \
  --s=${s} \
