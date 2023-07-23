import os
import sys
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))) )

import math
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from datetime import datetime
import os

from core.setting import config

def tblog_scalar(writer, step, root_name, **kwargs):
    for tag, value in kwargs.items():
        writer.add_scalar(root_name + '/' + tag, value, step)

def tblog_scalars(writer, step, root_name, **kwargs):
    for main_tag, value in kwargs.items():
        if isinstance(value, dict):
            writer.add_scalars(root_name + '/' + main_tag, value, step)
        else: raise Exception('scalars is not dictionary')

def tblog_image(writer, step, epoch, **kwargs): # note that input image<value> shape must be (3, H, W)
    freq = 'epoch' + str(epoch) + 'step' + str(step)
    for tag, value in kwargs.items():
        if len(value.shape) == 4:
            writer.add_images(freq + '/' + tag, value, step, dataformats='NCHW')
        elif len(value.shape) == 3:
            writer.add_image(freq + '/' + tag, value, step)

def tblog_embedding(writer, step, tag=None, mat=None, metadata=None, label_img= None):
    if len(mat.shape) > 2:
        mat = mat.reshape(mat.shape[0], -1)
    
    writer.add_embedding(mat=mat, metadata=metadata, label_img=label_img, global_tep=step, tag=tag)


# make log folder, tblog writer and model folder
def mk_logdir(log_root, log_name, config):
    log_path = os.path.join(log_root, log_name) + ' Date-' + datetime.today().strftime("%Y-%m-%d %H-%M-%S")

    if os.path.exists(log_path):
        print('log path is already exists.')
        raise ValueError
    else:
        os.mkdir(log_path)
        model_path = os.path.join(log_path, 'model')
        os.mkdir(model_path)

        # save config file
        config_path = os.path.join(log_path, 'config')
        os.mkdir(config_path)
        import pickle as pk
        with open(os.path.join(config_path, 'config.pk'), 'wb') as f:
            pk.dump(config, f)
        # make config txt
        mk_config_txt(config, config_path)
        
        writer = SummaryWriter(log_dir=os.path.join(log_path, 'tblog'))
    
        return writer, log_path, model_path

def mk_config_txt(config, config_path):
    var_name_list = dir(config)
    attr_name_list = []
    attr_list = []

    for var_name in var_name_list:
        if var_name[0] == '_':
            pass
        else: attr_name_list.append(var_name)

    for attr_name in attr_name_list:
        attr = getattr(config, attr_name)
        if str(type(attr)) == "<class 'method'>":
            attr_list.append(str(attr()))
        else:
            attr_list.append(str(attr))
    
    with open(os.path.join(config_path, 'config.txt'), 'w') as f:
        for i in range(len(attr_list)):
            f.write(attr_name_list[i] + ': ' + attr_list[i] + '\n')

# using code
# tensorboard --logdir=./

