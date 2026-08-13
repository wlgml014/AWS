# When Test-Time Adaptation Meets Self-Supervised Models

[![arXiv](https://img.shields.io/badge/arXiv-2506.23529-b31b1b.svg)](https://arxiv.org/abs/2506.23529)

This repository provides the implementation of test-time adaptation methods for self-supervised models on corrupted datasets. The experiments are conducted on CIFAR-10-C, CIFAR-100-C, and ImageNet-C datasets. This document outlines the setup process, environment requirements, and instructions to reproduce the experiments.
## Environment
Please create and activate the following conda environment.
```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate aws
```

## Setup
### Dataset Structure
Ensure that your dataset directory follows the structure below before running the experiments.
```text
cifar/
└── data/
    ├── CIFAR-10-C/
    └── CIFAR-100-C/
imagenet/
└── data/
    └── ImageNet-C/
```
### Model Checkpoints

This repository does not redistribute pretrained model checkpoints. Download them from the original repositories listed below and place them in the corresponding directories.

#### Supervised CIFAR Checkpoints

The supervised source models for CIFAR-10 and CIFAR-100 are provided by the [ViDA repository](https://github.com/Yangsenqiao/vida). Download the source-model checkpoints linked under the `Cifar10-to-Cifar10C task` and `Cifar100-to-Cifar100C task` sections of the ViDA README.

#### Self-Supervised Checkpoints

The MoCo and iBOT checkpoints used in this project are available in the `Supervised and Self-supervised Checkpoints` section of the [HiDe-Prompt repository](https://github.com/thu-ml/HiDe-Prompt). Download the MoCo and iBOT1k checkpoints.


If necessary, rename the downloaded files to `moco.pth` and `ibot1k.pth`. The same checkpoints may be copied or symbolically linked into both the `cifar/ckpt/` and `imagenet/ckpt/` directories. Please refer to the original repositories for their checkpoint licenses and terms of use.

#### Prototype Generation

The prototype files are not included in this repository. Generate both the full-data and 30-shot prototypes for each backbone (`dino`, `moco`, and `ibot1k`) using the scripts below. The `--model_opt` argument selects the backbone, and `--file` specifies the complete output path, including the `.npy` filename.

The paths passed to `--file` are resolved relative to the directory from which the command is run. Therefore, after running `cd cifar`, `ckpt_proto/cifar10/dino_full.npy` is saved as `cifar/ckpt_proto/cifar10/dino_full.npy`. Likewise, after running `cd imagenet`, `ckpt_proto/dino_full.npy` is saved as `imagenet/ckpt_proto/dino_full.npy`. Create the output directories before running the scripts and use the filenames shown below consistently.

For CIFAR-10 and CIFAR-100, run the following commands from the `cifar/` directory. The CIFAR datasets are downloaded automatically into `cifar/data/` if they are not already present.

```bash
cd cifar
mkdir -p ckpt_proto/cifar10 ckpt_proto/cifar100

# CIFAR-10: full-data prototypes
python prototype_cifar10.py --model_opt dino --file ckpt_proto/cifar10/dino_full.npy
python prototype_cifar10.py --model_opt moco --file ckpt_proto/cifar10/moco_full.npy
python prototype_cifar10.py --model_opt ibot1k --file ckpt_proto/cifar10/ibot1k_full.npy

# CIFAR-10: 30-shot prototypes
python prototype_cifar10_fs.py --model_opt dino --file ckpt_proto/cifar10/dino_30shot.npy
python prototype_cifar10_fs.py --model_opt moco --file ckpt_proto/cifar10/moco_30shot.npy
python prototype_cifar10_fs.py --model_opt ibot1k --file ckpt_proto/cifar10/ibot1k_30shot.npy

# CIFAR-100: full-data prototypes
python prototype_cifar100.py --model_opt dino --file ckpt_proto/cifar100/dino_full.npy
python prototype_cifar100.py --model_opt moco --file ckpt_proto/cifar100/moco_full.npy
python prototype_cifar100.py --model_opt ibot1k --file ckpt_proto/cifar100/ibot1k_full.npy

# CIFAR-100: 30-shot prototypes
python prototype_cifar100_fs.py --model_opt dino --file ckpt_proto/cifar100/dino_30shot.npy
python prototype_cifar100_fs.py --model_opt moco --file ckpt_proto/cifar100/moco_30shot.npy
python prototype_cifar100_fs.py --model_opt ibot1k --file ckpt_proto/cifar100/ibot1k_30shot.npy
```

For ImageNet, first set `imagenet_root` in both `imagenet/prototype.py` and `imagenet/prototype_fs.py` to the local ImageNet directory containing the `train/` and `val/` subdirectories. Then run the following commands from the `imagenet/` directory:

```bash
cd imagenet
mkdir -p ckpt_proto

# Full-data prototypes
python prototype.py --model_opt dino --file ckpt_proto/dino_full.npy
python prototype.py --model_opt moco --file ckpt_proto/moco_full.npy
python prototype.py --model_opt ibot1k --file ckpt_proto/ibot1k_full.npy

# 30-shot prototypes
python prototype_fs.py --model_opt dino --file ckpt_proto/dino_30shot.npy
python prototype_fs.py --model_opt moco --file ckpt_proto/moco_30shot.npy
python prototype_fs.py --model_opt ibot1k --file ckpt_proto/ibot1k_30shot.npy
```

The full-data scripts average features from all training images in each class. The few-shot scripts use 30 training images per class. Each saved `.npy` file contains the class prototypes and the prototype-classifier validation accuracy.

#### Directory Structure

After downloading the model checkpoints and generating the prototype files, the directories should have the following structure:

```text
cifar/
├── ckpt/
│   ├── cifar10.t7
│   ├── pretrain_cifar100.t7
│   ├── moco.pth
│   └── ibot1k.pth
└── ckpt_proto/
    ├── cifar10/
    │   ├── dino_full.npy
    │   ├── dino_30shot.npy
    │   ├── moco_full.npy
    │   ├── moco_30shot.npy
    │   ├── ibot1k_full.npy
    │   └── ibot1k_30shot.npy
    └── cifar100/
        ├── dino_full.npy
        ├── dino_30shot.npy
        ├── moco_full.npy
        ├── moco_30shot.npy
        ├── ibot1k_full.npy
        └── ibot1k_30shot.npy
imagenet/
├── ckpt/
│   ├── moco.pth
│   └── ibot1k.pth
└── ckpt_proto/
    ├── dino_full.npy
    ├── dino_30shot.npy
    ├── moco_full.npy
    ├── moco_30shot.npy
    ├── ibot1k_full.npy
    └── ibot1k_30shot.npy
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
### Cifar100-to-Cifar100C task 
```bash
cd cifar
bash ./bash/cifar100/aws_vit.sh
```

## Acknowledgements

This repository builds upon and uses components from the following open-source projects:

- This codebase is built upon the [official ViDA repository](https://github.com/Yangsenqiao/vida). The CIFAR-10 and CIFAR-100 supervised source checkpoints are also provided through ViDA.
- The CoTTA implementation is based on the [official CoTTA repository](https://github.com/qinenergy/cotta).
- The TENT implementation is based on the [official TENT repository](https://github.com/DequanWang/tent).
- The augmentation code uses components from the [KATANA repository](https://github.com/giladcohen/KATANA).
- Dataset and model utilities are based on [RobustBench](https://github.com/RobustBench/robustbench).
- The pretrained MoCo and iBOT checkpoints were obtained through the [HiDe-Prompt repository](https://github.com/thu-ml/HiDe-Prompt).

We thank the authors of these projects for making their code and pretrained models publicly available.
