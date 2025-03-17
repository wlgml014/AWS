from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
import cotta_transforms as my_transforms
from time import time
import logging


def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    #img_shape = (518, 518, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):#, iteration):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class EMA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()    

    def forward(self, x, y):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, y, self.model, self.optimizer)

        return outputs
        #     outputs, aug_mask, confidence_sample = self.forward_and_adapt(x, self.model, self.optimizer)

        # return outputs, aug_mask, confidence_sample
        #     teacher_outputs, student_outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        # return teacher_outputs, student_outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # use this line if you want to reset the teacher model as well. Maybe you also 
        # want to del self.model_ema first to save gpu memory.
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)                         


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, y, model, optimizer):
        outputs = self.model(x)
        self.model_ema.train()
        standard_ema = self.model_ema(x)
        
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=0.999)
        
        
        # Teacher Prediction
        standard_ema = self.model_ema(x)
        
        return standard_ema
    
    # @torch.enable_grad()  # ensure grads in possible no grad context for testing
    # def forward_and_adapt(self, x, model, optimizer):
    #     outputs = self.model(x)
    #     self.model_ema.train()
        
    #     # Teacher Prediction
    #     anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
    #     standard_ema = self.model_ema(x)
        
    #     # augmentation을 적용할 샘플에 대한 mask (anchor_prob < 0.1)
    #     aug_mask = anchor_prob < 0.1  # shape: [batch_size]
    #     confidence_sample = (anchor_prob<0.1).sum().item()
    #     N = 32  # augmentation 반복 횟수
        
    #     if aug_mask.any():
    #         # augmentation 대상 샘플만 선택하여 augmented prediction 계산
    #         outputs_emas = []
    #         for i in range(N):
    #             # x[aug_mask]에 대해 transform 적용 후 model_ema 실행
    #             aug_output = self.model_ema(self.transform(x[aug_mask])).detach()
    #             outputs_emas.append(aug_output)
    #         # augmentation한 결과들의 평균 (shape: [num_aug_samples, ...])
    #         augmented_avg = torch.stack(outputs_emas).mean(0)
            
    #         # 전체 배치의 예측값을 standard_ema로 초기화한 후,
    #         # augmentation 대상 샘플만 augmented_avg로 대체
    #         outputs_ema = standard_ema.clone()
    #         outputs_ema[aug_mask] = augmented_avg
    #     else:
    #         outputs_ema = standard_ema
        
    #     # Student update
    #     loss = (softmax_entropy(outputs, outputs_ema.detach())).mean(0) * 0
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
        
    #     # Teacher update
    #     self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=0.999)
        
    #     # Stochastic restore
    #     for nm, m in self.model.named_modules():
    #         for npp, p in m.named_parameters():
    #             if npp in ['weight', 'bias'] and p.requires_grad:
    #                 mask_param = (torch.rand(p.shape) < 0.001).float().cuda()
    #                 with torch.no_grad():
    #                     p.data = self.model_state[f"{nm}.{npp}"] * mask_param + p * (1.0 - mask_param)
        
    #     # 최종 출력과 함께 augmentation이 적용된 샘플에 대한 마스크도 return
    #     return outputs_ema, aug_mask, confidence_sample

@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
