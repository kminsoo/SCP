file_name="cifar100_slimming_finetune.sh"
name=$1
lr_init=$2
percent=$3
scale_init=$4
network=$5
gpu_number=$6
sudo docker run --runtime=nvidia --rm -it --name=${name} -v $PWD:/enas -w=/enas tensorflow/tensorflow:1.10.1-gpu /bin/bash ./scripts/${file_name} ${lr_init} ${percent} ${scale_init} ${network} ${gpu_number}
