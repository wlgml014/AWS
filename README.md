# When Test-Time Adaptation Meets Self-Supervised Model
This repository provides the implementation of test-time adaptation methods for self-supervised models on corrupted datasets. The experiments are conducted on CIFAR-10-C, CIFAR-100-C, and ImageNet-C datasets. This document outlines the setup process, environment requirements, and instructions to reproduce the experiments.
## Envorinments
Please create and activate the following conda envrionment. 
```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate aws
```

## Setup
### Dataset Structure
Ensure that your dataset directory follows the structure below before running the experiments.
```bash
cifar
├── data
│   ├── CIFAR-10-C
│   └── CIFAR-100-C
imagenet
└── ImageNet-C
```
### Model Checkpoints
The following structure should be used to store model checkpoints and prototype features.
```bash
cifar
├── ckpt
│   ├── cifar10.t7
│   ├── pretrain_cifar100.t7
│   ├── moco.pth
│   └── ibot1k.pth
├── ckpt_proto
│   ├── cifar10
│   │   ├── dino_full.npy
│   │   ├── dino_30shot.npy
│   │   ├── moco_full.npy
│   │   └──  ...
│   ├── cifar100
│   │   ├── dino_full.npy
│   │   ├── dino_30shot.npy
│   │   ├── moco_full.npy
│   │   └──  ...
imagenet
├── ckpt
│   ├── moco.pth
│   └── ibot1k.pth
├── ckpt_proto
│   ├── dino_full.npy
│   ├── dino_30shot.npy
└── └── ...
```
## Running Classification Experiments
This section provides instructions for running experiments on different datasets using test-time adaptation.
### ImageNet-to-ImageNetC task 
```bash
cd imagenet
bash ./bash/aws.sh
```
### Cifar10-to-Cifar10C task 
```bash
cd cifar
bash ./bash/cifar10/aws_vit.sh
```
## Cifar100-to-Cifar100C task 
```bash
cd cifar
bash ./bash/cifar100/aws_vit.sh
```
