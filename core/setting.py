from io import TextIOWrapper
import os

import torch

class config(object):
    # preprocessing
    dataset_root = './dataset'
    dataset = 'LSUN_bedroom'
    dataset_num = 50000 # variable
    dataset_name = dataset + '_' + str(dataset_num)
    dataset_path = os.path.join(dataset_root, dataset_name)
    preprocessing_option = 'load_path' 

    # data_loader
    batch_size = 1000
    num_workers = 0 # half number of cpu core is appropriate or x4 of gpus

    # global
    epoch = 100000
    iter = 0
    data_shape = [3, 64, 64]

    device = torch.device("cuda:0")
    gpu1 = torch.device("cuda:0")
    gpu2 = torch.device("cuda:1")
    cpu = torch.device('cpu')
    
    parallel = False
    lr_decay = False
    """
    device setting tip
    1. multi gpu
        device = torch.device("cuda")
        parallel = True
    
    2. single gpu
        device = torch.device("cuda:[GPU NUM])
        parallel = False
    """

    # model
        # generator
    g_name = 'dc_generator'
    g_lr = 0.0001
    latent_shape = [128, 1, 1]

        # discriminator
    d_name = 'dc_discriminator'
    d_lr = 0.0001
    use_attention = False # check
        
        # auxiliary classifier
    label_option = 'single_label'

        # loss 
        # NOTICE: All coef is variable
            # adv_loss
    adv_loss_name = 'ns_loss'
    coef_adv = 1

            # r1_reg_loss
    coef_r1 = 10

            # gaussian_reg_loss
    coef_gaussian = 0.2

            # mode_reg_loss
    coef_mode_adv = 0.5
    coef_img_dis = 0.5
    coef_mode = 1

            # perceptual_path_reg_loss
    coef_ppr = 0
    mean_path = None
            # aug_cls_loss
    coef_aux_cls = 0 # check

    # visualize log
    log_root = './logs'
    def mk_name(self):
        return self.dataset_name + '-' + self.memo + ' Batch-' + str(self.batch_size)
    metric_log_cycle = 1 
    img_log_cycle = 100
    fid_log_cycle = 100

    # save and load
    save_cycle = 100
    fid_path = os.path.join(dataset_root, 'FID', dataset + '.pk')
    min_fid = float('inf')

    # atc
    memo = 'exp_52 augmentaiton d output DCGAN' # check

def open_config():
    None