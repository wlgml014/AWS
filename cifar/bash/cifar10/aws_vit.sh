export PYTHONPATH=.
conda deactivate
conda activate aws

# supervised setting
python cifar10c_vit.py --cfg cfgs/cifar10/aws.yaml --model_opt sl --batch 16 --lr 0.000025
# selfsupervised setting
## dino
### full-shot
python cifar10c_vit.py --cfg cfgs/cifar10/aws.yaml --model_opt dino --batch 16 --lr 0.0000025
### few-shot
python cifar10c_vit.py --cfg cfgs/cifar10/aws.yaml --model_opt dino --batch 16 --lr 0.0000025 --shot few

## moco
### full-shot
python cifar10c_vit.py --cfg cfgs/cifar10/aws.yaml --model_opt moco --batch 16 --lr 0.0000025
### few-shot
python cifar10c_vit.py --cfg cfgs/cifar10/aws.yaml --model_opt moco --batch 16 --lr 0.0000025 --shot few

## ibot1k
### full-shot
python cifar10c_vit.py --cfg cfgs/cifar10/aws.yaml --model_opt ibot1k --batch 16 --lr 0.00000025
### few-shot
python cifar10c_vit.py --cfg cfgs/cifar10/aws.yaml --model_opt ibot1k --batch 16 --lr 0.00000025 --shot fews