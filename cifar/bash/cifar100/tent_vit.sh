export PYTHONPATH=.
conda deactivate
conda activate aws

python cifar100c_vit.py --cfg cfgs/cifar100/tent.yaml --checkpoint /path/to/checkpoint.pth --data_dir /path/to/datasets