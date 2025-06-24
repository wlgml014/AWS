export PYTHONPATH=.
conda deactivate
conda activate aws

python cifar10c_vit.py --cfg cfgs/cifar10/cotta.yaml --checkpoint [your-ckpt-path] --data_dir [your-data-dir] 