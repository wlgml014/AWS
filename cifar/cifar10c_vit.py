import logging
import os
import timm
import numpy as np
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F


from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy
import torchvision.transforms as transforms

import tent
import cotta
import aws
import sar

from ssl_model import get_dino, get_moco, get_ibot1k

import torch.nn as nn
from conf import cfg, load_cfg_from_args

logger = logging.getLogger(__name__)


def evaluate(description):
    args = load_cfg_from_args(description)
    # configure model
    
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
    
    # for cotta threshold
    ap = 0.1
    
    if cfg.SHOT is not None:
        if cfg.SHOT == "full":
            fs = False
        elif cfg.SHOT == "few":
            fs = True
    
    if sl:
        ckpt_path = "./ckpt/cifar10.t7"
        base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                        cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
        checkpoint = torch.load(ckpt_path)
        new_state_dict = {}
        for key, value in checkpoint["model"].items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        base_model.load_state_dict(new_state_dict, strict=True)
        base_model = base_model.cuda()
        ap = 0.92
    
    if dino:
        logger.info("get dino")
        base_model = get_dino()
        if fs:
            prototype_file = "./ckpt_proto/cifar10/dino_30shot.npy"
        else:
            prototype_file = "./ckpt_proto/cifar10/dino_full.npy"
        ap = 0.4077    
    if moco:
        logger.info("get moco")
        base_model = get_moco()
        base_model = base_model.cuda()
        if fs:
            prototype_file = "./ckpt_proto/cifar10/moco_30shot.npy"
        else:
            prototype_file = "./ckpt_proto/cifar10/moco_full.npy"
        ap = 0.1521
    if ibot1k:
        logger.info("get ibot1k")
        base_model = get_ibot1k()
        base_model = base_model.cuda()
        if fs:
            prototype_file = "./ckpt_proto/cifar10/ibot1k_30shot.npy"
        else:
            prototype_file = "./ckpt_proto/cifar10/ibot1k_full.npy"
        ap=0.4154
    
    if not sl:
        if os.path.exists(prototype_file):
            data = np.load(prototype_file, allow_pickle=True).item()
            prototypes_array = data["prototypes"]
        else:
            raise FileNotFoundError(f"There is no {prototype_file}.")
        
        base_model = replace_classifier_with_prototype(base_model, prototypes=prototypes_array)
        base_model = base_model.cuda()
    
    # size
    if sl:
        resize = 384
        crop = 384
        args.size = 384
    else:
        resize = 256
        crop = 224
        args.size = 224
        
    # normalzie
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model, ap=ap)
    if cfg.MODEL.ADAPTATION == "aws":
        logger.info("test-time adaptation: AWS")
        model = setup_aws(base_model, cfg.MODEL_OPT)
    if cfg.MODEL.ADAPTATION == "sar":
        logger.info(f"test-time adaptation: SAR")
        model = setup_sar(base_model)
    
    # evaluate on each severity and type of corruption in turn
    All_error = []
    
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            if i_c == 0:
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test = x_test.cuda()
            y_test = y_test.cuda()
            x_test = torch.nn.functional.interpolate(x_test, size=(cfg.SIZE, cfg.SIZE), \
                mode="bilinear", align_corners=False)
            
            # normalize
            if not sl:
                normalize = transforms.Normalize(mean=mean, std=std)
                x_test = normalize(x_test)
            
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, device = 'cuda')
            err = 1. - acc
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


def setup_aws(model, model_opt="sl"):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = aws.configure_model(model)
    params, params_ssl = aws.collect_params(model)
    optimizer = setup_optimizer_aws(params, params_ssl, cfg.OPTIM.LR, cfg.OPTIM.LR*10) #1e-4 , 1e-2
    aws_model =aws.AWS(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, model_opt=model_opt)
    logger.info(f"model for adaptation: %s", aws_model)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return aws_model

def setup_cotta(model, ap=0.92):
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
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           #ap=cfg.OPTIM.AP,
                           ap = ap,
                           size = cfg.SIZE)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

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
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError
    
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
    evaluate("CIFAR-10-C evaluation.")
