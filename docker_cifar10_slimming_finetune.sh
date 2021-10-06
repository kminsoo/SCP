file_name="cifar10_slimming_finetune.sh"
name=$1
percent=$2
network=$3
gpu_number=$4
sudo docker run --runtime=nvidia --rm -it --name=${name} -v $PWD:/enas -w=/enas tensorflow/tensorflow:1.10.1-gpu /bin/bash ./scripts/${file_name} ${percent} ${network} ${gpu_number}
