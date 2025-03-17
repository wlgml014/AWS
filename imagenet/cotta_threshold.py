import logging
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
from conf import cfg, load_cfg_fom_args
from ssl_model import get_dino, get_dinov2, get_moco, get_ibot1k, get_ibot21k
import torch.nn.functional as F

seeds = [1]
logger = logging.getLogger(__name__)
imagenet_root = "/mnt/data01/jihee/data/ImageNet"
train_dir = os.path.join(imagenet_root, "train")
val_dir = os.path.join(imagenet_root, "val")

class PrototypeClassifier(torch.nn.Module):
    def __init__(self, in_features, num_classes, prototypes=None):
        super(PrototypeClassifier, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_classes, in_features))
        if prototypes is not None:
            if isinstance(prototypes, np.ndarray):
                prototypes = torch.tensor(prototypes, dtype=torch.float32)
            self.weight.data = torch.nn.functional.normalize(prototypes, p=2, dim=1).cuda()
        else:
            torch.nn.init.constant_(self.weight, 1.0)
            self.weight.data = torch.nn.functional.normalize(self.weight.data, p=2, dim=1).cuda()

    def forward(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        #return 40*torch.matmul(x, self.weight.t())
        return 40 * torch.matmul(x, F.normalize(self.weight.data, p=2, dim=1).t())
        
def replace_classifier_with_prototype(model, prototypes=None):
    num_classes = prototypes.shape[0] if prototypes is not None else cfg.MODEL.NUM_CLASSES
    classifier_attr = None
    in_features = None
    #print(model)
    if hasattr(model, "fc"):
        classifier_attr = "fc"
        in_features = model.fc.in_features
    elif hasattr(model, "head"):
        classifier_attr = "head"
        if isinstance(model.head, torch.nn.Linear):
            in_features = model.head.in_features
        else:
            for module in reversed(model.head):
                if isinstance(module, torch.nn.Linear):
                    in_features = module.in_features
                    break
    else:
        raise ValueError("no attribute fc or head")

    new_classifier = PrototypeClassifier(in_features, num_classes, prototypes)
    setattr(model, classifier_attr, new_classifier)
    logger.info(f"Replaced {classifier_attr} to PrototypeClassifier: in_features={in_features}, num_classes={num_classes}")
    return model

dino = True
dinov2 = False
sl = False
openclip = False
moco = False
ibot1k = False
ibot21k = False

# CIFAR10용 모델 로드  
args = load_cfg_fom_args()
fs = False
if sl:
    #ckpt_path = "./ckpt/vit_base_384_cifar10.t7"
    ckpt_path = "./ckpt/pretrain_cifar100.t7"
    model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                    cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
    #if cfg.TEST.ckpt is not None:
    if True:
        print(1)
        #base_model = torch.nn.DataParallel(base_model) # make parallel
        checkpoint = torch.load(ckpt_path)
        new_state_dict = {}
        for key, value in checkpoint['model'].items():
            # backbone.blocks.~ -> blocks.~
            new_key = key.replace("module.", "")  # "backbone."을 제거하여 모델과 일치시킴
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=True)
    else:
        model = torch.nn.DataParallel(model) # make parallel
    model = model.cuda()
    del checkpoint
    del new_state_dict

    resize = 384
    crop = 384
if dino:
    logger.info("get dino")
    model = get_dino()
    prototype_file = "/mnt/data01/jihee/AWS/imagenet/ckpt_proto/imagenet_class_prototypes_fullshot_seed1.npy"
    if os.path.exists(prototype_file):
        data = np.load(prototype_file, allow_pickle=True).item()
        prototypes_array = data["prototypes"]
    else:
        raise FileNotFoundError(f"There is no {prototype_file}.")
    
    model = replace_classifier_with_prototype(model, prototypes=prototypes_array)
    model = model.cuda()
    model.eval()
    size = 224
    resize = 256
    crop = 224
if moco:
    logger.info("get moco")
    model = get_moco()
    if False:
        prototype_file = "/mnt/data01/jihee/AWS/cifar/ckpt_proto/cifar10/moco_class_prototypes_full_seed1.npy"
    else:
        prototype_file = "/mnt/data01/jihee/AWS/imagenet/ckpt_proto/moco_imagenet_class_prototypes_full_seed1.npy"
        
    if os.path.exists(prototype_file):
        data = np.load(prototype_file, allow_pickle=True).item()
        prototypes_array = data["prototypes"]
    else:
        raise FileNotFoundError(f"There is no {prototype_file}.")
    
    model = replace_classifier_with_prototype(model, prototypes=prototypes_array)
    model = model.cuda()
    model.eval()
    size = 224
    resize = 256
    crop = 224
if ibot1k:
    logger.info("get ibot1k")
    model = get_ibot1k()
    if False:
        prototype_file = "/mnt/data01/jihee/AWS/cifar/ckpt_proto/cifar10/ibot1k_class_prototypes_full_seed1.npy"
    else:
        prototype_file = "/mnt/data01/jihee/AWS/imagenet/ckpt_proto/iBOT_imagenet_class_prototypes_full_seed1.npy"
    if os.path.exists(prototype_file):
        data = np.load(prototype_file, allow_pickle=True).item()
        prototypes_array = data["prototypes"]
    else:
        raise FileNotFoundError(f"There is no {prototype_file}.")
    
    model = replace_classifier_with_prototype(model, prototypes=prototypes_array)
    model = model.cuda()
    model.eval()
    size = 224
    resize = 256
    crop = 224
    
if ibot21k:
    logger.info("get ibot21k")
    model = get_ibot21k()
    prototype_file = "/mnt/data01/jihee/AWS/imagenet/ckpt_proto/ibot21k_imagenet_class_prototypes_full_seed1.npy"
    if os.path.exists(prototype_file):
        data = np.load(prototype_file, allow_pickle=True).item()
        prototypes_array = data["prototypes"]
    else:
        raise FileNotFoundError(f"There is no {prototype_file}.")
    
    model = replace_classifier_with_prototype(model, prototypes=prototypes_array)
    model = model.cuda()
    model.eval()
    size = 224
    resize = 256
    crop = 224
    

        
    
if False:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

if False:
    mean =  [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]        
    
transform = transforms.Compose([
    transforms.Resize(resize),
    transforms.CenterCrop(crop),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

dataset = datasets.ImageFolder(train_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)

for seed in seeds:
    print(f"\nProcessing for seed {seed}...\n")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    correct = 0
    total = 0
    all_confidences = []  # confidence 값 저장 리스트

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating Seed {seed}"):
            images, labels = images.cuda(), labels.cuda()
            # classification head를 포함하여 모델의 예측값(logits)을 구함
            outputs = model(images)
            # softmax를 적용하여 각 클래스의 확률 계산
            probs = torch.nn.functional.softmax(outputs, dim=1)
            # 각 이미지별 최대 확률(신뢰도)와 예측 label 계산
            confidences, predictions = probs.max(dim=1)
            # accuracy 계산
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            # confidence 값 누적
            all_confidences.extend(confidences.cpu().numpy())

    accuracy_val = correct / total * 100
    lower_5_conf = np.percentile(np.array(all_confidences), 5)
    print(f"Validation Accuracy for seed {seed}: {accuracy_val:.2f}%")
    print(f"Lower 5% confidence value for seed {seed}: {lower_5_conf:.4f}")
