file_name="cifar100_finetune.sh"
name=$1
lr_init=$2
network=$3
load_dir=$4
gpu_number=$5
logistic_c=$6
iteration=$7
sudo docker run --runtime=nvidia --rm -it --name=${name} -v $PWD:/enas -w=/enas tensorflow/tensorflow:1.10.1-gpu /bin/bash ./scripts/${file_name} ${lr_init} ${network} ${load_dir} ${gpu_number} ${logistic_c} ${iteration}
