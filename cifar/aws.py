from copy import deepcopy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import torchvision.transforms as transforms
import random

class AWS(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, model_opt="sl", num_class=10):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.model_opt = model_opt
        self.num_class = num_class
        
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
           copy_model_and_optimizer(self.model, self.optimizer)
        self.ssl_model = copy_model(self.model)
        
    def forward(self, x, y=None):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, y, self.model, self.ssl_model, self.optimizer, self.model_state)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, y, model, ssl_model, optimizer, model_state=None):
        """Forward and adapt model on batch of data.

        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        outputs = model(x)
        
        model.eval()
        features = model.forward_features(x)[:,0,:]
        ssl_features = ssl_model.forward_features(x)[:,0,:].detach()
        model.train()

        outputs_ssl = model.head_ssl(ssl_features)
        # adapt
        
        if self.num_class == 10:
            loss, positive_mask, negative_mask = pseudo_label_contrastive_loss(features, outputs, k=1, n=5, temperature=0.5)
        if self.num_class == 100:
            loss, positive_mask, negative_mask = pseudo_label_contrastive_loss(features, outputs, k=1, n=2, temperature=0.5)
        
        if self.model_opt == "sl":
            lamda1 = 0.01
            lamda2 = 0.4
            loss += lamda1 * relatation_kd_loss(features, ssl_features, positive_mask)
        else:
            lamda1 = 0.5
            lamda2 = 0.4
            loss += lamda1 * kd_loss(features, ssl_features)   
        
        loss += lamda2 * iid_loss(outputs, outputs_ssl.detach())
        pseudo_labels = torch.argmax(outputs, dim=1)
        loss += lamda2 * nn.CrossEntropyLoss()(outputs_ssl, pseudo_labels).mean()
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
                                
        return outputs

def kd_loss(features, ssl_features, temperature=2):
    features = F.normalize(features, dim=1)
    ssl_features = F.normalize(ssl_features, dim=1)
    kd_loss = torch.mean(torch.frobenius_norm(features - ssl_features, dim=-1))
    return kd_loss


def relatation_kd_loss(features, ssl_features, positive_mask, temperature=2):
    tta_model_similarities = sim_matrix(features, features)
    ssl_model_similarities = sim_matrix(ssl_features, ssl_features)

    rkd_loss = nn.MSELoss()(features, ssl_features)

    return rkd_loss.mean()

def sim_matrix(a, b, eps=1e-8):
    """
    Batch cosine similarity taken from https://stackoverflow.com/a/58144658/10425618
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def pseudo_label_contrastive_loss(features, outputs, k=5, n=20, temperature=0.5):
    """
    Computes contrastive loss using Top-K pseudo-labels.
    Encourages similarity if there is a shared top-K class, and dissimilarity otherwise.

    Args:
        features (torch.Tensor): Feature embeddings (B, D).
        outputs (torch.Tensor): Model logits (B, num_classes).
        k (int): Number of top classes for pseudo-labeling.
        temperature (float): Temperature for contrastive loss.

    Returns:
        loss (torch.Tensor): Contrastive loss encouraging both similarity and dissimilarity.
    """
    batch_size = features.shape[0]

    # Get top-K pseudo-labels
    top_k_classes, _ = topk_pseudo_labels(outputs, k)
    top_n_classes, _ = topk_pseudo_labels(outputs, n)

    # Normalize feature vectors
    features = F.normalize(features, dim=1)

    # Compute cosine similarity matrix (B, B)
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # Create positive and negative masks
    labels_matrix = top_k_classes[:, None, :] == top_k_classes[None, :, :]
    positive_mask = labels_matrix.any(dim=2).float().to(features.device)  # (B, B)
    labels_matrix = top_n_classes[:, None, :] == top_n_classes[None, :, :]
    positive_mask_n = labels_matrix.any(dim=2).float().to(features.device)  # (B, B)
    negative_mask = (1 - positive_mask_n).float().to(features.device)  # Inverse of positive mask

    # Compute log probabilities
    exp_sim = torch.exp(similarity_matrix)
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))  # (B, B)

    # Compute positive and negative losses
    positive_loss = -(positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
    negative_loss = (negative_mask * similarity_matrix).sum(dim=1) / (negative_mask.sum(dim=1) + 1e-8)

    # Combine losses (balancing factor can be adjusted if needed)
    contrastive_loss = positive_loss + negative_loss

    return contrastive_loss.mean(), positive_mask, negative_mask
    
def topk_pseudo_labels(output, k=5):
    """
    Returns Top-K pseudo-labels and their probabilities.
    
    Args:
        output (torch.Tensor): Logits from model (B, num_classes).
        k (int): Number of top classes to consider.

    Returns:
        top_k_classes (torch.Tensor): Indices of top-k classes (B, k).
        top_k_probs (torch.Tensor): Normalized probabilities (B, k).
    """
    probs = F.softmax(output, dim=1)  # Convert logits to probabilities
    top_k_probs, top_k_classes = probs.topk(k, dim=1)  # Get top-k probs and indices
    top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)  # Normalize to sum 1
    return top_k_classes, top_k_probs

def iid_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
      x_out = F.softmax(x_out,dim=-1)
      x_tf_out = F.softmax(x_tf_out,dim=-1)
      _, k = x_out.size()
      bn_, k_ = x_out.size()
      assert (x_tf_out.size(0) == bn_ and x_tf_out.size(1) == k_)
      su_temp1 = x_out.unsqueeze(2)
      su_temp2 = x_tf_out.unsqueeze(1)
      p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
      p_i_j = p_i_j.sum(dim=0)  # k, k
      p_i_j = (p_i_j + p_i_j.t()) / 2.
      p_i_j = p_i_j / p_i_j.sum()
      assert (p_i_j.size() == (k, k))

      p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
      p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)
      p_i_j[(p_i_j < EPS).data] = EPS

      loss = - p_i_j * (torch.log(p_i_j) \
                        - lamb * torch.log(p_j) \
                        - lamb * torch.log(p_i))

      loss = loss.sum()
      return loss


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    ssl_params_list = []
    model_params_lst = []
    for name, param in model.named_parameters():
        if 'ssl' in name:
            ssl_params_list.append(param)
        else:
            if "head" not in name:
                model_params_lst.append(param)
    return model_params_lst, ssl_params_list

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def copy_model(model):
    """Copy the model and optimizer states for resetting after adaptation."""
    #model_state = deepcopy(model.state_dict())
    ssl_model = deepcopy(model)
    #optimizer_state = deepcopy(optimizer.state_dict())
    #ema_model = deepcopy(model)
    for param in ssl_model.parameters():
        param.detach_()
    return ssl_model
    
def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    model.head_ssl = deepcopy(model.head)
    # disable grad, to (re-)enable only what tent updates
    # model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
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
