## Operation-Aware Soft Channel Pruning using Differentiable Masks
Repository for Operation-Aware Soft Channel Pruning using Differentiable Masks (ICML 2020)

#### 1. Dependencies
This code is implemented based on [TensorFlow Docker](https://www.tensorflow.org/install/docker?hl=ko) with version 1.10.1-gpu and python 2.
The algorithm is tested on Ubuntu 16.04.


#### 2. CIFAR-10 Dataset Download
If you already have CIFAR-10 dataset, please change the location as follows:
```bash
mv "your CIFAR-10 directory location" ./data
mv ./data/* ./data/cifar10
```
Otherwise, you can download it as follows:
```bash
wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
mv cifar-10-batches-py ./data
mv ./data/* ./data/cifar10
```
#### 3. Experiment
You can test baseline with several networks as following command:
ex) DenseNet
```bash
sudo bash docker_cifar10_base.sh DenseNet DenseNet 0
```
If you want to run other networks, you just replace "DenseNet" with "VggNet19", "VggNet16", and "ResNet".
Then, training outputs are saved in ./cifar10_DenseNet_base_network in case of "DenseNet".

Also, please run the following command to reproduce our algorithm with several networks.
```bash
# ResNet
sudo bash docker_cifar10_gumbel_prune.sh ResNet_gumbel_prune 0.95 0.00005 2.0 0 ResNet
# DenseNet
sudo bash docker_cifar10_gumbel_prune.sh DenseNet_gumbel_prune 0.95 0.00003 2.0 0 DenseNet
# VggNet19
sudo bash docker_cifar10_gumbel_prune.sh VggNet19_gumbel_prune 0.95 0.0001 2.0 0 VggNet19
# VggNet16
sudo bash docker_cifar10_gumbel_prune.sh VggNet16_gumbel_prune 0.95 0.0001 2.0 0 VggNet16
```
The training outputs are saved in ./cifar10_ResNet_gumbel_prune in case of "ResNet".

Finally, you can check "Slimming" algorithm as following command:
ex) DenseNet
```bash
# Learning a network from scratch with a sparse regularization defined in "Slimming".
sudo bash docker_cifar10_slimming.sh DenseNet_Slimming DenseNet 0
```
If you want to run other networks, you just replace "DenseNet" with "VggNet19" and "VggNet16".
After training, the training outputs are saved in ./cifar10_DenseNet_slimming_network.
To prune and then fine-tune with "Slimming", you can run the following command.
ex) DenseNet
```bash
# Pruning the pre-trained network for 80% channels and then fine-tune the pruned network
sudo bash docker_cifar10_slimming_finetune.sh DenseNet_Slimming_Finetune 0.8 DenseNet 0
```
The training outputs are saved in ./cifar10_DenseNet_slimming_finetune_0.8.
If you want to run other networks, you just replace "DenseNet" with "VggNet19" and "VggNet16".
Also, you can prune 90% channels in a network by changing 0.8 into 0.9.

### Citation
```
@inproceedings{kang2020operation,
  title="{Operation-Aware Soft Channel Pruning using Differentiable Masks}",
  author={Kang, Minsoo and Han, Bohyung},
  booktitle={ICML},
  year={2020},
}
```

