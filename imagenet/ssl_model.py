import os
import torch
import timm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
import numpy as np


def get_dino():
    model = load_model("Standard_VITB", "./ckpt",
                       "imagenet", ThreatModel.corruptions)
    dino_model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
    dino_state_dict = dino_model.state_dict()
    model.load_state_dict(dino_state_dict, strict=False)
    
    model = model.cuda()
    return model

def get_sl():
    model = load_model("Standard_VITB", "./ckpt",
                       "imagenet", ThreatModel.corruptions)
    model.eval()
    model = model.cuda()
    return model

def get_moco():
    model = load_model("Standard_VITB", "./ckpt",
                       "imagenet", ThreatModel.corruptions)
    checkpoint = torch.load("./ckpt/moco.pth", map_location="cpu")

    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        # backbone.blocks.~ -> blocks.~
        new_key = key.replace("module.base_encoder.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=False)
    model = model.cuda()
    return model

def get_ibot1k():
    model = load_model("Standard_VITB", "./ckpt",
                       "imagenet", ThreatModel.corruptions)
    checkpoint = torch.load("./ckpt/ibot1k.pth", map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()
    return model
