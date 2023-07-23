import os
import sys
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))) )

import torch
import torch.nn as nn
from torchvision.transforms.functional import resize
from torchvision.models import inception_v3
from torch import linalg as la

from core.model import Generator, VGG
from core.setting import config
from core.func import load_model
from metric.fid_my import *

"""필요한것
1. 훈련된 모델
2. 특징 추출 모델

순서도
1. 훈련된 모델에서 이미지 생성
2. 생성 이미지의 해상도를 특징 추출 모델의 해상도에 맞춰 조정
3. 특징 추출
4. 특징의 평균과 분산 계산 """

cfg = config()

def feat_extract(generator, extractor, batch_size, img_shape, device):
    z = torch.randn([batch_size] + cfg.latent_shape, device=device)
    gen_img = generator(z)
    resized_gen_img = resize(gen_img, img_shape)
    img_feat = extractor(resized_gen_img)[0]
    return img_feat

def cal_feat_var(generator, feat_extractor, batch_size, feat_num, img_shape, device): # propoer input image shape for feature extractor (H, W)
    iter = feat_num // batch_size
    last_batch = feat_num % batch_size
    feat_list = []

    with torch.no_grad():
        for i in range(iter):
            feat_list.append(feat_extract(generator, feat_extractor, batch_size, img_shape, device))

        feat_list.append(feat_extract(generator, feat_extractor, last_batch, img_shape, device))

        feat = torch.cat(feat_list, dim=0)
        # std mean 이런식으로 구하면 안될듯
        if len(feat.shape) == 2:
            mean_feat = torch.unsqueeze(torch.mean(torch.transpose(feat, 0, 1), dim=1), dim=0)
            dis_mat = feat - mean_feat
        elif len(feat.shape) == 4:
            mean_feat = torch.unsquueze(torch.mean(feat.permute(1,2,3,0), dim=3), dim=0)
            dis_mat = torch.flatten(feat - mean_feat, start_dim=1)
        
        l2_dis = la.norm(dis_mat, ord=2, dim=1)
        
        stdv = torch.std(l2_dis)

    
    return stdv, mean_feat.detach()

if __name__ == '__main__':
    gen_path = None
    device ='cuda'
    dim = 2048
    batch_size = 100
    feat_num = 10000
    img_shape = (299, 299)

    generator = Generator().to(device)
    feat_extractor = def_extractor(device=device, dim=dim)
    load_model(generator, gen_path)

    with torch.no_grad():
        for i in range(10):
            stdv, mean_feat = cal_feat_var(generator, feat_extractor, batch_size, feat_num, img_shape, device)
            print(stdv)




