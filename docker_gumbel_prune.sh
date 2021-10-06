file_name="cifar100_gumbel_prune.sh"
name=$1
output_dir=$2
sparse_threshold=$3
logistic_k=$4
logistic_c=$5
logistic_c2=$6
logistic_c3=$7
temperature=$8
scale_weight=$9
sparse_bernoulli=${10}
sparse_bernoulli2=${11}
sparse_bernoulli3=${12}
gpu_number=${13}
use_L1=${14}
mask_reg=${15}
network=${16}
use_uniform_noise=${17}
minval=${18}
maxval=${19}

sudo docker run --runtime=nvidia --rm -it --name=${name} -v $PWD:/enas -w=/enas tensorflow/tensorflow:1.10.1-gpu /bin/bash ./scripts/${file_name} ${output_dir} ${sparse_threshold} ${logistic_k} ${logistic_c} ${logistic_c2} ${logistic_c3} ${temperature} ${scale_weight} ${sparse_bernoulli} ${sparse_bernoulli2} ${sparse_bernoulli3} ${gpu_number} ${use_L1} ${mask_reg} ${network} ${use_uniform_noise} ${minval} ${maxval}
