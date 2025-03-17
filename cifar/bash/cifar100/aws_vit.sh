export PYTHONPATH=.
conda deactivate
conda activate vida

# supervised setting

# selfsupervised setting
## dino
### full-shot
python cifar100c_vit.py --cfg ./cfgs/cifar100/aws.yaml --model_opt dino --batch 16 --lr 0.0000025
### few-shot
python cifar100c_vit.py --cfg ./cfgs/cifar100/aws.yaml --model_opt dino --batch 16 --lr 0.0000025 --shot few

## moco
### full-shot
python cifar100c_vit.py --cfg ./cfgs/cifar100/aws.yaml --model_opt moco --batch 16 --lr 0.0000025
### few-shot
python cifar100c_vit.py --cfg ./cfgs/cifar100/aws.yaml --model_opt moco --batch 16 --lr 0.0000025 --shot few

## ibot1k
### full-shot
python cifar100c_vit.py --cfg ./cfgs/cifar100/aws.yaml --model_opt ibot1k --batch 16 --lr 0.00000025
### few-shot
python cifar100c_vit.py --cfg ./cfgs/cifar100/aws.yaml --model_opt ibot1k --batch 16 --lr 0.00000025 --shot few