file_name="cifar100_base.sh"
name=$1
network=$2
gpu_number=$3
sudo docker run --runtime=nvidia --rm -it --name=${name} -v $PWD:/enas -w=/enas tensorflow/tensorflow:1.10.1-gpu /bin/bash ./scripts/${file_name} ${network} ${gpu_number}
