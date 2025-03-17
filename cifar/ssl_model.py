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
    # model = load_model("Standard_VITB", "./ckpt",
    #                    "imagenet", ThreatModel.corruptions)
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    #model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
    dino_model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
    dino_state_dict = dino_model.state_dict()
    model.load_state_dict(dino_state_dict, strict=False)
    
    model = model.cuda()
    return model

def get_sl():
    model = timm.create_model('vit_base_patch16_384', pretrained=True)
    # model = load_model("Standard_VITB", "./ckpt",
    #                    "imagenet", ThreatModel.corruptions)
    #model.eval()
    model = model.cuda()
    return model

def get_moco():
    
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    #checkpoint = torch.load("/mnt/data01/jihee/AWS/imagenet/ckpt/vit-b-300ep.pth")
    checkpoint = torch.load("./ckpt/moco.pth")
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        # backbone.blocks.~ -> blocks.~
        new_key = key.replace("module.base_encoder.", "")  # "backbone."을 제거하여 모델과 일치시킴
        #new_key = key.replace("module.momentum_encoder.", "")  # "backbone."을 제거하여 모델과 일치시킴
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False)
    #model.eval()
    model = model.cuda()

    return model

def get_ibot1k():
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    checkpoint = torch.load("./ckpt/ibot1k.pth")
    #checkpoint = torch.load("/mnt/data01/jihee/AWS/imagenet/ckpt/ibot1k.pth", map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()
    return model
