file_name="cifar10_gumbel_prune.sh"
name=$1
logistic_c=$2
sparse_bernoulli=$3
s=$4
gpu_number=$5
network=$6
sudo docker run --runtime=nvidia --rm -it --name=${name} -v $PWD:/enas -w=/enas tensorflow/tensorflow:1.10.1-gpu /bin/bash ./scripts/${file_name} ${logistic_c} ${sparse_bernoulli} ${s} ${gpu_number} ${network}
