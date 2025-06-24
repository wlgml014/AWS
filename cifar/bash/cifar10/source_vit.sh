export PYTHONPATH=.
conda deactivate
conda activate aws

python cifar10c_vit.py --cfg cfgs/cifar10/source.yaml --checkpoint ckpt/vit_base_384_cifar10.t7