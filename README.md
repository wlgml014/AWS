# When Test-Time Adaptation Meets Self-Supervised Model

## Envorinments
Please create and activate the following conda envrionment. 
```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate aws
```

## Setup
```bash
cifar
├── data
│   ├── CIFAR-10-C
│   └── CIFAR-100-C
imagenet
└── ImageNet-C
```
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
## Classification Experiments
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
