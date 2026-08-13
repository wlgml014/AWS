import logging
import os
import timm
import numpy as np
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from robustbench.data import load_imagenetc, load_imagenet
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model, load_model_bayes
from robustbench.utils import clean_accuracy as accuracy


import tent
import cotta
import aws
import sar
from ssl_model import get_dino, get_moco, get_ibot1k


from conf import cfg, load_cfg_from_args


logger = logging.getLogger(__name__)



def evaluate(description):
    args = load_cfg_from_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()


    sl = False
    dino = False
    moco = False
    ibot1k = False

    
    
    if cfg.MODEL_OPT is not None:
        if cfg.MODEL_OPT == "dino":
            dino = True
        elif cfg.MODEL_OPT == "moco":
            moco = True
        elif cfg.MODEL_OPT == "ibot1k":
            ibot1k = True
        elif cfg.MODEL_OPT == "sl":
            sl = True

    ap = 0.1
    
    if cfg.SHOT is not None:
        if cfg.SHOT == "full":
            fs = False
        elif cfg.SHOT == "few":
            fs = True
    #fs = False
    
    if dino:
        logger.info("get dino")
        base_model = get_dino()
        if fs:
            prototype_file = "./ckpt_proto/dino_30shot.npy"
        else:
            prototype_file = "./ckpt_proto/dino_full.npy"
        ap = 0.3391
    if moco:
        logger.info("get moco")
        base_model = get_moco()
        if fs:
            prototype_file = "./ckpt_proto/moco_30shot.npy"
        else:
            prototype_file = "./ckpt_proto/moco_full.npy"
        ap = 0.016
    if ibot1k:
        logger.info("get ibot1k")
        base_model = get_ibot1k()
        if fs:
            prototype_file = "./ckpt_proto/ibot1k_30shot.npy"
        else:
            prototype_file = "./ckpt_proto/ibot1k_full.npy"
        ap = 0.3848

    
    if not sl:
        if os.path.exists(prototype_file):
            data = np.load(prototype_file, allow_pickle=True).item()
            prototypes_array = data["prototypes"]
        else:
            raise FileNotFoundError(f"There is no {prototype_file}.")
        
        base_model = replace_classifier_with_prototype(base_model, prototypes=prototypes_array)
        base_model = base_model.cuda()

    # size
    resize = 256
    crop = 224
    
    # normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    
    if sl:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    elif cfg.MODEL.ADAPTATION == "tent":
        logger.info(f"test-time adaptation: TENT")
        model = setup_tent(base_model)
    elif cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model, ap=ap)
    elif cfg.MODEL.ADAPTATION == "aws":
        logger.info("test-time adaptation: AWS")
        model = setup_aws(base_model, cfg.MODEL_OPT)
    elif cfg.MODEL.ADAPTATION == "sar":
        logger.info(f"test-time adaptation: SAR")
        model = setup_sar(base_model)
    else:
        raise ValueError(
            f"Unsupported adaptation method: {cfg.MODEL.ADAPTATION}"
        ) 
        

    # evaluate on each severity and type of corruption in turn
    prev_ct = "x0"
    topk = 5
    All_error = []
    for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
        for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # reset adaptation for each combination of corruption x severity
            # note: for evaluation protocol, but not necessarily needed
            try:
                if i_x == 0:
                    model.reset()
                    logger.info("resetting model")
                else:
                    logger.warning("not resetting model")
            except:
                logger.warning("not resetting model")
            
            x_test, y_test = load_imagenetc(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type], transform=transform)
            
            x_test, y_test = x_test.cuda(), y_test.cuda()

            
            acc, acc_topk= accuracy(model, x_test, y_test, topk, cfg.TEST.BATCH_SIZE)
            
            err = 1. - acc
            err_topk = 1. - acc_topk
            All_error.append(err)

            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
    mean_err = sum(All_error)/len(All_error)
    logger.info(f"Mean error: {mean_err:.2%}")

def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    
    logger.info(f"model for evaluation: %s", model)
    return model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_sar(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    sar_model = sar.SAR(model, params, lr = cfg.OPTIM.LR, 
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", sar_model.optimizer)
    return sar_model



def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    else:
        raise NotImplementedError

def setup_cotta(model, ap=0.1):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           ap = ap)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def setup_aws(model, model_opt = "sl"):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = aws.configure_model(model)
    params, params_ssl = aws.collect_params(model)
    optimizer = setup_optimizer_aws(params, params_ssl, cfg.OPTIM.LR, cfg.OPTIM.LR*10) #1e-4 , 1e-4 //1e-2
    aws_model =aws.AWS(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, model_opt=model_opt)
    logger.info(f"model for adaptation: %s", aws_model)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)
    return aws_model

def setup_optimizer_aws(params, params_ssl, model_lr, ssl_lr):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD([{"params": params, "lr": model_lr},
                          {"params": params_ssl, "lr": ssl_lr}],
                   lr=cfg.OPTIM.LR,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    else:
        raise NotImplementedError

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

if __name__ == '__main__':
    evaluate("ImageNet-C evaluation.")
