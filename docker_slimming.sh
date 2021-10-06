file_name="cifar100_slimming.sh"
name=$1
scale_init=$2
network=$3
gpu_number=$4
sudo docker run --runtime=nvidia --rm -it --name=${name} -v $PWD:/enas -w=/enas tensorflow/tensorflow:1.10.1-gpu /bin/bash ./scripts/${file_name} ${scale_init} ${network} ${gpu_number}
