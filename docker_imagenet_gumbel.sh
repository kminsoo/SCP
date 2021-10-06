file_name="train_imagenet_gumbel.sh"
name=$1
output_dir=$2
gpu_number=$3
num_gpus=$4
network=$5
sparse_threshold=$6
logistic_k=$7
logistic_c=$8
logistic_c2=$9
temperature=${10}
sparse_bernoulli=${11}
sparse_bernoulli2=${12}

sudo docker run --runtime=nvidia --rm -it --name=${name} -v $PWD:/enas -v /imagenet:/imagenet -w=/enas tensorflow/tensorflow:1.10.1-gpu /bin/bash ./scripts/${file_name} ${output_dir} ${gpu_number} ${num_gpus} ${network} ${sparse_threshold} ${logistic_k} ${logistic_c} ${logistic_c2} ${temperature} ${sparse_bernoulli} ${sparse_bernoulli2}
