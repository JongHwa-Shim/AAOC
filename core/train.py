import os
import sys
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))) )
import time
from torchvision.models import resnet50

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt
from torch.nn import DataParallel
from torch.optim import Adam
from torchvision import transforms

from core.build_dataloader import *
from core.setting import config
from core.model import *
from core.func import *
from core.augmentation import aug
import core.augmentation as augmentation
from metric.loss import *
from visualize.tblog import *
from metric.fid_my import *

import pdb

aug = aug()
cfg = config()
cfg.pre_aug_func_list = aug.pre_aug_func_list
cfg.aug_func_list = aug.aug_func_list
cfg.aug_num = len(aug.aug_func_list)


def train():
    # make save folder and tblog
    writer, log_path, model_path = mk_logdir(cfg.log_root, cfg.mk_name(), cfg)

    # preprocessing
    preprocessed_data = preprocessing(cfg.dataset_path, cfg.preprocessing_option)

    # build dataloader
    transform = my_transform(compose_transform(option=cfg.preprocessing_option))
    dataset = SimpleDataset(preprocessed_data, transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers = 4, shuffle=True, pin_memory=True)

    preprocessed_data_test = preprocessing('./dataset/CelebA_10000', cfg.preprocessing_option)
    dataset_test = SimpleDataset(preprocessed_data_test, transform)
    dataloader_test = DataLoader(dataset_test, batch_size=cfg.batch_size, shuffle=False)

    # define model
    G = to_device(DC_Generator(), cfg.device, parallel=cfg.parallel)
    D = to_device(DC_Discriminator(), cfg.device, parallel=cfg.parallel)
    """img_encoder = to_device(Image_encoder(), cfg.device, parallel=cfg.parallel)
    feat_extractor = to_device(VGG(), cfg.device, parallel=cfg.parallel)"""

    """model_list = [G, D, img_encoder, feat_extractor]"""
    model_list = [G, D]

    # define optimizer
    """G_opt = Adam(list(G.parameters()) + list(img_encoder.parameters()), lr=cfg.g_lr)"""
    G_opt = Adam(list(G.parameters()), lr=cfg.g_lr)
    D_opt = Adam(D.parameters(), lr=cfg.d_lr)

    opt_list = [G_opt, D_opt] # for learning rate decay

    # define additional things
    z_fixed = torch.randn([25] + cfg.latent_shape, device=cfg.device)
    inception = def_extractor(device='cpu', dim=2048) # inceptionv3 for fid

    # training
    for epoch in range(cfg.epoch):
        # learning rate decay
        if cfg.lr_decay:
            lr_decay([G_opt, D_opt], epoch, per_epoch=500, decay_rate=0.1)
        start = 0
        for i, data in enumerate(dataloader):
            print('iter time is ' + str(time.time() - start))
            # prepare data
            start = time.time()

            cfg.batch_size = data.shape[0]
            real_img = data.to(cfg.device)
            """g_in = torch.randn([cfg.batch_size] + cfg.g_input_shape, device=cfg.device)"""
            z_fake_for_D = torch.randn([cfg.batch_size] + cfg.latent_shape, device=cfg.device)
            z_fake_for_G = torch.randn([cfg.batch_size] + cfg.latent_shape, device=cfg.device, requires_grad=True)

            print('data time is ' + str(time.time() - start))

            # Discriminator
                # real_img -> aug -> D
            start = time.time()

            with torch.no_grad():
                real_aug_img, aug_task_real = aug.augmentation(real_img)
            """D_real, aux_gap_real, aux_gmp_real, real_heat_map = D(real_aug_img, teacher_aug_task=aug_task_real)"""
            D_real, aux_real = D(real_aug_img, aug_task_real)
            
            D_real_adv_loss = adv_loss(D_real, 1) * cfg.coef_adv
            ## D_real_aux_loss = aug_cls_loss(aux_real, aug_task_real) * cfg.coef_aux_cls
            """D_real_aux_loss = (aug_cls_loss(aux_gap_real, aug_task_real) + aug_cls_loss(aux_gmp_real, aug_task_real)) * cfg.coef_aux_cls"""

                # z_fake -> G -> aug -> D
            with torch.no_grad():
                fake_img = G(z_fake_for_D)
                #fake_img, _, _, _, _ = G(z_fake_for_D)
                fake_aug_img, aug_task_fake = aug.augmentation(fake_img)
            """D_fake_for_D, aux_gap_fake, aux_gmp_fake, fake_heat_map = D(fake_aug_img, teacher_aug_task=aug_task_fake)"""
            D_fake_for_D, aux_fake_for_D = D(fake_aug_img, aug_task_fake)

            D_fake_adv_loss = adv_loss(D_fake_for_D, 0) * cfg.coef_adv
            ## D_fake_aux_loss = aug_cls_loss(aux_fake_for_D, aug_task_fake) * cfg.coef_aux_cls
            """D_fake_aux_loss = (aug_cls_loss_adv(aux_gap_fake, aug_task_fake) + aug_cls_loss_adv(aux_gmp_fake, aug_task_fake)) * cfg.coef_aux_cls"""
            
                # real_img -> img_encoder -> interpolate(z_real, z_fake) -> G -> aug -> D (Plan1, deprecated)
                # img_encoder is sucks in first point.
                # interpolate(fake_img, real_img) -> aug -> D (Plan2)
                    # Plan 1
            """with torch.no_grad():
                z_real = img_encoder(real_img)
                z_fake = torch.randn([real_img.shape[0], cfg.style_dim], device=cfg.device)
                gamma = torch.FloatTensor(cfg.batch_size, 1).uniform_(0,1).to(cfg.device)
                intrpl_z = gamma * z_real + (1 - gamma) * z_fake
                intrpl_aug_img, aug_task_intrpl = aug(G(g_in, intrpl_z))
            intrpl_aug_img.requires_grad = True
            D_intrpl, aux_intrpl, intrpl_heat_map = D(intrpl_aug_img, teacher_aug_task=aug_task_intrpl)
            r1_loss = r1_reg_loss(D_intrpl, intrpl_aug_img) * cfg.coef_r1"""
            
                    # Plan 2
            
            """with torch.no_grad():
                gamma = torch.FloatTensor(real_img.shape[0], 1, 1, 1).uniform_(0,1).to(cfg.device)
                intrpl_img = gamma * real_aug_img + (1 - gamma) * fake_aug_img
                #intrpl_aug_img, aug_task_intrpl = aug(intrpl_img)
                intrpl_aug_img = intrpl_img
            intrpl_aug_img.requires_grad = True
            D_intrpl = D(intrpl_aug_img)
            #D_intrpl, aux_gap_intrpl, aux_gmp_intrpl, intrpl_heat_map = D(intrpl_aug_img, teacher_aug_task=aug_task_intrpl)
            r1_loss = r1_reg_loss(D_intrpl, intrpl_aug_img) * cfg.coef_r1"""
            

                # calculate final D loss
            """D_loss = D_real_adv_loss + D_fake_adv_loss + D_real_aux_loss + D_fake_aux_loss + r1_loss"""
            D_loss = D_real_adv_loss + D_fake_adv_loss
            D_opt.zero_grad()
            D_loss.backward()
            D_opt.step()

            print('D time is ' + str(time.time() - start))

            # Generator
                # z_fake -> G -> aug -> D
            start = time.time()
            fake_img= G(z_fake_for_G)
            #fake_img, _1, _2, _3, _4 = G(z_fake_for_G)
            fake_aug_img, aug_task_fake = aug.augmentation(fake_img)
            """D_fake, aux_gap_fake, aux_gmp_fake, fake_heat_map = D(fake_aug_img, teacher_aug_task=aug_task_fake)"""
            D_fake, aux_fake = D(fake_aug_img, aug_task_fake)

            G_fake_adv_loss = adv_loss(D_fake, 1) * cfg.coef_adv
            ## G_fake_aux_loss = aug_cls_loss(aux_fake, aug_task_fake) * cfg.coef_aux_cls
            """G_fake_aux_loss = (aug_cls_loss(aux_gap_fake, aug_task_fake) + aug_cls_loss(aux_gmp_fake, aug_task_fake)) * cfg.coef_aux_cls"""
            
            """ppr_loss, cfg.mean_path, path_length, mean_grad = perceptual_path_reg_loss(fake_img, z_fake_for_G, feat_extractor, cfg.mean_path)
            ppr_loss = ppr_loss * cfg.coef_ppr"""
            
                # real_img -> img_encoder -> G -> aug -> D
            """z_real = img_encoder(real_img)
            reconst_img = G(g_in, z_real)
            reconst_aug_img, aug_task_reconst = aug(reconst_img)
            D_reconst, aux_gap_reconst, aux_gmp_reconst, reconst_heat_map = D(reconst_aug_img, teacher_aug_task=aug_task_reconst)

            mode_loss = mode_reg_loss(fake_img, real_img, D_reconst, cfg.coef_img_dis, cfg.coef_mode_adv) * cfg.coef_mode # reconst loss + adv loss
            gaussian_latent_loss = gaussian_reg_loss(z_real) * cfg.coef_gaussian"""

                # calculate final G_loss            
            """G_loss = G_fake_adv_loss + G_fake_aux_loss + ppr_loss #+ mode_loss + gaussian_latent_loss"""
            G_loss = G_fake_adv_loss
            G_opt.zero_grad()
            G_loss.backward()
            G_opt.step()

            print('G time is ' + str(time.time() - start))

            start = time.time()

            # visualize
            # for scalar data
            if cfg.iter%cfg.metric_log_cycle == 0:
                # prepare scalar
                D_real_out, D_fake_out = eval_overfitting(D_real, D_fake_for_D)
                
                scalar_dict = {'D_real_adv_loss': D_real_adv_loss,
                            'D_fake_adv_loss': D_fake_adv_loss, 'G_fake_adv_loss': G_fake_adv_loss}
                scalars_dict = {'D_out': {'D_real': D_real_out, 'D_fake_out': D_fake_out}}
                
                # log scalar
                tblog_scalar(writer, step=cfg.iter, root_name='metrics', **scalar_dict)
                tblog_scalars(writer, step=cfg.iter, root_name='metrics', **scalars_dict)

            # for image data
            if cfg.iter % cfg.img_log_cycle == 0:
                # prepare image
                """
                with torch.no_grad():
                    fake_img_fixed = G(z_fixed)
                    #fake_img_fixed, _, _, _, _ = G(z_fixed)
                """
                
                # for test_data in dataloader_test:
                #     test_data = test_data.to(cfg.device)
                #     left_rotate = augmentation.left_rotation(test_data)
                #     right_rotate = augmentation.right_rotation(test_data)
                #     horizontal_rotate = augmentation.horizontal_rotation(test_data)

                #     with torch.no_grad():
                #         left_out, _ = D(left_rotate, None)
                #         right_out, _ = D(right_rotate, None)
                #         horizontal_out, _ = D(horizontal_rotate, None)
                #         out, _ = D(test_data, None)

                #         left_out = left_out.mean()
                #         right_out = right_out.mean()
                #         horizontal_out = horizontal_out.mean()
                #         out = out.mean()

                #         d_out_dict = {'D_aug_out': {'90 degree': left_out, '180 degree': horizontal_out, '270 degree': right_out, '360 degree': out}}
                #         tblog_scalars(writer, step=cfg.iter, root_name='metrics', **d_out_dict)

                img_dict = {'generated_img': minmax_scale(fake_img[0:25], max=1, min=-1, new_max=1, new_min=0), 
                            'real_aug_img': minmax_scale(real_aug_img[0:25], max=1, min=-1, new_max=1, new_min=0)}
                tblog_image(writer, step=cfg.iter, epoch=epoch, **img_dict)

            print('result time is ' + str(time.time() - start))
            # save
            """
            if cfg.save_cycle is not False and cfg.iter % cfg.save_cycle == 0:
                start = time.time()
                with torch.no_grad():
                    z_fixed = torch.randn([1000] + cfg.latent_shape, device=cfg.device)
                    fake_img_fid = G(z_fixed)
                fid = calculate_fid_tensor_real_time(cfg.fid_path, fake_img_fid, inception)
                tblog_scalar(writer, step=cfg.iter, root_name='metrics', **{'FID': fid})
                print('fid time is ' + str(time.time()-start))
                if fid < cfg.min_fid:
                    None
                    #save_model(model_list, epoch, cfg.iter, parallel=cfg.parallel, path=model_path)
            """

                    

            cfg.iter += 1
            start = time.time()

    a = 1
if __name__=='__main__':
    train()