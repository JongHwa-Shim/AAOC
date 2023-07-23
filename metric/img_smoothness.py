import os
import sys
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))) )

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import resize
from torchvision.models import inception_v3

from core.setting import config
from core.model import Generator
from core.func import load_model
from metric.fid_my import *

cfg = config()

def eval_smoothness(generator, feat_extractor, step_size, img_shape, device): # img_shape: proper input image shape for feature extractor (H, W)
    interpl_step = step_size - 1 # 1을 빼줘야 딱 맞음
    z_0 = torch.zeros([1] + cfg.latent_shape, device=device)
    z_6sig = (torch.ones([1] + cfg.latent_shape, device=device) * 6) # 1 텐서가 아니라 norm이 1인 텐서로 바꿔줄 필요

    grad_norm_list = []
    for num in range(interpl_step+1):
        gamma = num/interpl_step
        interpl_z = ((1 - gamma) * z_0 + gamma * z_6sig).requires_grad_(True)

        gen_img = generator(interpl_z)
        resized_gen_img = resize(gen_img, img_shape)
        img_feat = feat_extractor(resized_gen_img)[0]

        gradient = torch.autograd.grad(outputs=img_feat.sum(), inputs=interpl_z, create_graph=False, retain_graph=False, only_inputs=True)[0]
        grad_norm = torch.sqrt(gradient.view(-1).pow(2).mean(0) + 1e-12)
        grad_norm_list.append(grad_norm.clone().detach())
    
    return grad_norm_list

if __name__ == '__main__':
    gen_path = ''
    save_path = './results/image_smoothness'
    name = ''
    device = 'cuda'
    dim=2048
    img_shape = (299, 299)
    step_size = 1000

    generator = Generator().to(device)
    feat_extractor = def_extractor(device=device, dim=dim)
    load_model(generator, gen_path)
    
    grad_norm_list = eval_smoothness(generator, feat_extractor, step_size, img_shape, device)

    writer = SummaryWriter(log_dir=os.path.join(save_path, name))

    for i, grad_norm in enumerate(grad_norm_list):
        writer.add_scalar('eval_smoothness', i, grad_norm)





        




    