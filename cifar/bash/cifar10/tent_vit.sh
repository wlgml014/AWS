export PYTHONPATH=.
conda deactivate
conda activate aws

python cifar10c_vit.py --cfg cfgs/cifar10/tent.yaml --checkpoint /path/to/checkpoint.pth --data_dir /path/to/datasets