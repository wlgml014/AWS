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
from conf import cfg, load_cfg_from_args
from ssl_model import get_dino, get_moco, get_ibot1k
import torch.nn.functional as F

seeds = [1]
logger = logging.getLogger(__name__)
imagenet_root = "/path/to/imagenet"
train_dir = os.path.join(imagenet_root, "train")
val_dir = os.path.join(imagenet_root, "val")

def replace_classifier_with_prototype(model, prototypes=None):
    num_classes = prototypes.shape[0] if prototypes is not None else cfg.MODEL.NUM_CLASSES
    classifier_attr = None
    in_features = None
    if hasattr(model, "fc"):
        classifier_attr = "fc"
        in_features = model.fc.in_features
    elif hasattr(model, "head"):
        classifier_attr = "head"
        if isinstance(model.head, torch.nn.Linear):
            in_features = model.head.in_features
        elif isinstance(model.head, torch.nn.Identity):
            if hasattr(model, "num_features"):
                in_features = model.num_features
            elif hasattr(model, "embed_dim"):
                in_features = model.embed_dim
            else:
                raise ValueError("model.head is Identity, but cannot infer feature dimension.")
        elif isinstance(model.head, torch.nn.Sequential):
            for module in reversed(model.head):
                if isinstance(module, torch.nn.Linear):
                    in_features = module.in_features
                    break
        else:
            try:
                for module in reversed(list(model.head.children())):
                    if isinstance(module, torch.nn.Linear):
                        in_features = module.in_features
                        break
            except TypeError:
                pass

            if in_features is None:
                if hasattr(model, "num_features"):
                    in_features = model.num_features
                elif hasattr(model, "embed_dim"):
                    in_features = model.embed_dim
                else:
                    raise ValueError(f"Cannot infer in_features from head type: {type(model.head)}")
    else:
        raise ValueError("no attribute fc or head")

    new_classifier = PrototypeClassifier(in_features, num_classes, prototypes)
    setattr(model, classifier_attr, new_classifier)
    logger.info(f"Replaced {classifier_attr} to PrototypeClassifier: in_features={in_features}, num_classes={num_classes}")
    return model


class PrototypeClassifier(torch.nn.Module):
    def __init__(self, in_features, num_classes, prototypes=None):
        super(PrototypeClassifier, self).__init__()

        self.weight = torch.nn.Parameter(torch.empty(num_classes, in_features))

        with torch.no_grad():
            if prototypes is not None:
                if isinstance(prototypes, np.ndarray):
                    prototypes = torch.tensor(prototypes, dtype=torch.float32)

                prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=1)
                self.weight.copy_(prototypes)
            else:
                torch.nn.init.constant_(self.weight, 1.0)
                self.weight.copy_(torch.nn.functional.normalize(self.weight, p=2, dim=1))

    def forward(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        weight = torch.nn.functional.normalize(self.weight, p=2, dim=1)

        return 40 * torch.matmul(x, weight.t())


dino = True
sl = False
moco = False
ibot1k = False

 
args = load_cfg_from_args()
fs = False
if sl:
    #ckpt_path = "./ckpt/vit_base_384_cifar10.t7"
    ckpt_path = "./ckpt/pretrain_cifar100.t7"
    model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                    cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
    #if cfg.TEST.ckpt is not None:
    if True:
        print(1)
        checkpoint = torch.load(ckpt_path)
        new_state_dict = {}
        for key, value in checkpoint['model'].items():
            new_key = key.replace("module.", "")  
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=True)
    else:
        model = torch.nn.DataParallel(model) 
    model = model.cuda()
    del checkpoint
    del new_state_dict

    resize = 384
    crop = 384
if dino:
    logger.info("get dino")
    model = get_dino()
    prototype_file = "./ckpt_proto/dino_full.npy"
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
    prototype_file = "./ckpt_proto/moco_full.npy"  
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
    prototype_file = "./ckpt_proto/ibot1k_full.npy"
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
    all_confidences = [] 

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating Seed {seed}"):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = probs.max(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_confidences.extend(confidences.cpu().numpy())

    accuracy_val = correct / total * 100
    lower_5_conf = np.percentile(np.array(all_confidences), 5)
    print(f"Validation Accuracy for seed {seed}: {accuracy_val:.2f}%")
    print(f"Lower 5% confidence value for seed {seed}: {lower_5_conf:.4f}")
