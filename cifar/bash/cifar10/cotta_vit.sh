export PYTHONPATH=.
conda deactivate
conda activate aws

python cifar10c_vit.py --cfg cfgs/cifar10/cotta.yaml --checkpoint /path/to/checkpoint.pth --data_dir /path/to/datasets