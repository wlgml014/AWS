export PYTHONPATH=
conda deactivate
conda activate aws

# supervised setting
python imagenetc.py --cfg ./cfgs/vit/aws.yaml --model_opt sl --batch 64 --lr 1e-4

# selfsupervised setting
## dino
### full-shot
python imagenetc.py --cfg ./cfgs/vit/aws.yaml --model_opt dino --batch 64 --lr 1e-4
### few-shot
python imagenetc.py --cfg ./cfgs/vit/aws.yaml --model_opt dino --batch 64 --lr 1e-4 --shot few

## moco
### full-shot
python imagenetc.py --cfg ./cfgs/vit/aws.yaml --model_opt moco --batch 64 --lr 1e-4
### few-shot
python imagenetc.py --cfg ./cfgs/vit/aws.yaml --model_opt moco --batch 64 --lr 1e-4 --shot few

## ibot1k
### full-shot
python imagenetc.py --cfg ./cfgs/vit/aws.yaml --model_opt ibot1k --batch 64 --lr 1e-4
### few-shot
python imagenetc.py --cfg ./cfgs/vit/aws.yaml --model_opt ibot1k --batch 64 --lr 1e-4 --shot few