file_name="train_imagenet.sh"
name=$1
output_dir=$2
gpu_number=$3
num_gpus=$4
network=$5

sudo docker run --runtime=nvidia --rm -it --name=${name} -v $PWD:/enas -v /imagenet:/imagenet -w=/enas tensorflow/tensorflow:1.10.1-gpu /bin/bash ./scripts/${file_name} ${output_dir} ${gpu_number} ${num_gpus} ${network}
