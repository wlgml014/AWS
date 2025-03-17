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
import norm
import cotta
import aws
import sar
import come
import petal

from ssl_model import get_dino, get_moco, get_ibot1k

import torch.nn as nn
from conf import cfg, load_cfg_fom_args

logger = logging.getLogger(__name__)


def evaluate(description):
    args = load_cfg_fom_args(description)
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
    #fs = False
    
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
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
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
    if cfg.MODEL.ADAPTATION == "come":
        logger.info("test-time adaptation: COME")
        model = setup_come(base_model)
    if cfg.MODEL.ADAPTATION == "petal":
        logger.info(f"test-time adaptation: PETAL")
        
        if sl:
            swa_model_path = "./ckpt/petal/cifar10/vit_swa.pt"
            cov_model_path = "./ckpt/petal/cifar10/vit_cov.pt"
        elif dino:
            swa_model_path = "./ckpt/petal/cifar10/dino_swa.pt"
            cov_model_path = "./ckpt/petal/cifar10/dino_cov.pt"
        elif moco:
            swa_model_path = "./ckpt/petal/cifar10/moco_swa.pt"
            cov_model_path = "./ckpt/petal/cifar10/moco_cov.pt"
        elif ibot1k:
            swa_model_path = "./ckpt/petal/cifar10/ibot1k.pt"
            cov_model_path = "./ckpt/petal/cifar10/ibot1k.pt"
        
        base_mean_model = copy.deepcopy(base_model)
        base_cov_model = copy.deepcopy(base_model)

        checkpoint_swa = torch.load(swa_model_path, map_location=torch.device('cpu'))
        checkpoint_cov = torch.load(cov_model_path, map_location=torch.device('cpu'))

        base_mean_model.load_state_dict(checkpoint_swa, strict=True)
        base_cov_model.load_state_dict(checkpoint_cov, strict=True)

        base_model = base_model.cuda()
        base_mean_model = base_mean_model.cuda()
        base_cov_model = base_cov_model.cuda()

        model = setup_petal(base_model, base_mean_model, base_cov_model, ap=ap)
    
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
    logger.info(f"Mean Err : {mean_err}")

def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


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

def setup_petal(model, mean_model, cov_model, ap=0.1):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = petal.configure_model(model)
    mean_model = petal.configure_model(mean_model)
    cov_model = petal.configure_model(cov_model)
    params, param_names = petal.collect_params(model)
    optimizer = setup_optimizer(params)
    spw = 1e-9
    perc = 0.03
    logger.info(f"spw & perc: {spw} , {perc}")
    petalfim_model = petal.PETALFim(model, mean_model, cov_model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           spw=spw, 
                           perc=perc,
                           ap=ap)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return petalfim_model

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


def setup_come(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = come.configure_model(model)
    params, param_names = come.collect_params(model)
    come_model = come.COME(model, params, cfg.OPTIM.LR,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", come_model.optimizer)
    return come_model


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

if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')
