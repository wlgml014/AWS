export PYTHONPATH=.
conda deactivate
conda activate aws

python cifar100c_vit.py --cfg cfgs/cifar100/source.yaml --checkpoint ckpt/pretrain_cifar100.t7 