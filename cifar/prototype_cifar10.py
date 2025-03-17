import argparse
import os
import torch
import timm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
import numpy as np
import random
from tqdm import tqdm

seeds = [1]


parser = argparse.ArgumentParser()

parser.add_argument("--model_opt", dest="model_opt", default="dino", type=str)
parser.add_argument("--file", dest="file", default="dino_full.npy", type=str)

args = parser.parse_args()
model_opt = args.model_opt
filename = args.file


dino = False
moco = False
ibot1k = False


if model_opt is not None:
    if model_opt == "dino":
        print("dino")
        dino = True
    elif model_opt == "moco":
        print("moco")
        moco = True
    elif model_opt == "ibot1k":
        print("ibot1k")
        ibot1k = True

if dino:
    model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
    model.eval()
    model = model.cuda()
if moco:
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    checkpoint = torch.load("./ckpt/moco.pth")
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace("module.base_encoder.", "") 
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model = model.cuda()
if ibot1k:
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    checkpoint = torch.load("./ckpt/ibot1k.pth", map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()

resize = 256
crop = 224

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 학습 데이터에 대한 전처리: 데이터 증강 (RandomCrop, RandomHorizontalFlip)와 정규화
transform = transforms.Compose([
    transforms.Resize(resize),
    transforms.CenterCrop(crop),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    
])



# CIFAR-10 학습 데이터셋 로드 (다운로드가 되어 있지 않으면 자동으로 다운로드)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# CIFAR-10 테스트 데이터셋 로드 (이를 validation 데이터로 사용)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader 생성: 배치 단위로 데이터를 로드 (여기서는 배치 크기를 128로 설정)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)


for seed in seeds:
    print(f"\nProcessing for seed {seed}...\n")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    class_features = {cls: [] for cls in range(len(train_dataset.classes))}

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc=f"Extracting Features (Seed {seed})"):
            images = images.cuda()
            features = model.forward_features(images)

            if isinstance(features, dict): 
                features = features["x_norm_clstoken"]
            else:
                features = features[:, 0, :]

            features = torch.nn.functional.normalize(features, p=2, dim=-1).cpu().numpy()

            for feat, label in zip(features, labels.numpy()):
                class_features[label].append(feat)

    prototypes = {cls: np.mean(class_features[cls], axis=0) for cls in class_features if len(class_features[cls]) > 0}

    prototypes_array = np.array([prototypes[cls] for cls in sorted(prototypes.keys())])
    prototypes_tensor = torch.tensor(prototypes_array, dtype=torch.float32).cuda()
    
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Evaluating Seed {seed}"):
            images, labels = images.cuda(), labels.cuda()
            features = model.forward_features(images)

            if isinstance(features, dict):
                features = features["x_norm_clstoken"]
            else:
                features = features[:, 0, :]

            features = torch.nn.functional.normalize(features, p=2, dim=1)

            similarities = torch.matmul(features, prototypes_tensor.T)  # (batch_size, num_classes)
            predictions = similarities.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Validation Accuracy for seed {seed}: {accuracy:.2f}%")

    save_data = {
        "prototypes": prototypes_array,
        "accuracy": accuracy
    }
    np.save(filename, save_data)
    print(f"Saved: {filename} (Shape: {prototypes_array.shape}, Accuracy: {accuracy:.2f}%)")
