import math

import torch
from torch.autograd import grad
import torch.nn.functional as F

from core.setting import config

cfg = config()

def adv_loss(input, target):
    target_tensor = torch.full_like(input, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(input, target_tensor) # non saturating loss
    #loss = F.mse_loss(F.sigmoid(input),target_tensor) # SEGAN
    """if target == 1:
        loss = -1 * input.mean()
    elif target == 0:
        loss = input.mean() # wgan loss"""
    return loss


def r1_reg_loss(d_out, x_in):
    batch = d_out.shape[0]
    gradient = torch.autograd.grad(outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient = gradient.pow(2)
    penalty = 0.5 * gradient.view(batch, -1).sum(1).mean(0)
    """gradient_norm = torch.sqrt(gradient.view(batch, -1).sum(1) + 1e-12)
    penalty = ((gradient_norm - 1) **2).mean() # WGAN-GP"""
    return penalty


def gaussian_reg_loss(latent_vec): # input is laten vector, shape = (N, 512)
    # gaussian mean = 0, gaussian std = 1
    mean = latent_vec.mean(1)
    mean_target = torch.full_like(mean, fill_value=0)
    mean_loss = F.mse_loss(mean, mean_target)

    std = torch.std(latent_vec, dim=1)
    std_target = torch.full_like(std, fill_value=1)
    std_loss = F.mse_loss(std, std_target)

    loss = mean_loss + std_loss
    return loss

def mode_reg_loss(input, target, d_out, coef_img_dis, coef_mode_adv):
    reconst_loss = F.l1_loss(input, target)
    mode_adv_loss = adv_loss(d_out, 1)
    return coef_img_dis * reconst_loss + coef_mode_adv * mode_adv_loss

def perceptual_path_reg_loss(g_out, z_in, feat_extractor, mean_path_length_list, decay=0.01):
    noise = torch.randn_like(g_out) / math.sqrt(g_out.shape[2] * g_out.shape[3])
    g_out = g_out * noise

    feature_maps = feat_extractor(g_out) # consider multiple vgg layers, num of feature maps are 2

    path_penalty = 0
    path_mean_list = []
    path_length_list = []
    grad_mean_list = []
    for i, feature in enumerate(feature_maps):
        grad = torch.autograd.grad(outputs=feature.sum(), inputs=z_in, create_graph=True, retain_graph=True, only_inputs=True)[0]
        path_length = torch.sqrt(grad.pow(2).sum(1).mean(0))

        if mean_path_length_list is None:
            path_mean = path_length
        else:
            path_mean = mean_path_length_list[i] + decay * (path_length - mean_path_length_list[i])
        path_penalty += (path_length - path_mean).pow(2).mean()

        path_mean_list.append(path_mean.detach())
        path_length_list.append(path_length.detach())
        grad_mean_list.append(torch.mean(grad))

    return path_penalty, path_mean_list, path_length_list, grad_mean_list

def aug_cls_loss(input, aug_task, label_option=cfg.label_option):
    if label_option == 'single_label': # single augmentation
        loss = F.cross_entropy(input, aug_task)
    elif label_option == 'multi_label': # cumulative augmentation
        loss = F.binary_cross_entropy_with_logits(input, aug_task)
    return loss

def _aug_cls_loss_adv(input, aug_task):
    input = F.softmax(input, dim=1) # (N,H)
    aug_task_fake = [torch.unsqueeze(input[i][aug_task[i]:aug_task[i]+1], dim=0) for i in range(cfg.batch_size)]
    aug_task_fake = torch.cat(aug_task_fake, dim=0) # (N, 1)
    target_tensor = torch.full_like(aug_task_fake, 0)
    loss = F.binary_cross_entropy(aug_task_fake, target_tensor)
    return loss


