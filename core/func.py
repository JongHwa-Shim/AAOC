import cv2 as cv
import numpy as np
import os


import torch
import torch.nn as nn

# for image processing
def open_image(input): # datatype = path
    bgr_image = cv.imread(input) # BGR, (H, W, C)
    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB).astype(np.float32) # RGB, (H, W, C)
    rgb_image = np.transpose(rgb_image, (2, 0, 1)) # RGB (C, H, W)
    return rgb_image

def numpy2tensor(input):
    return torch.from_numpy(input)

def minmax_scale(input, max=255, min=0, new_max=1, new_min=-1): # input: tensor
    if max == None and min == None:
        max = input.max()
        min = input.min()

    return (input - min)/(max - min) * (new_max - new_min) + new_min

def bypass(input):
    return input

# for training
def to_device(model, device, parallel=None):
    if parallel:
        if device == torch.device('cpu'):
            new_model = model.to(device)
        else:
            new_model = nn.DataParallel(model).to(device)
    else:
        new_model = model.to(device)
    
    return new_model

def lr_decay(opts, current_epoch, per_epoch=500, decay_rate=0.1):
    if not isinstance(opts, list):
        raise ValueError
    if current_epoch is not 0 and current_epoch % per_epoch is 0:
        for opt in opts:
            opt.param_groups[0]['lr'] *= (1-decay_rate)
    
# evalute discriminator overfitting
def eval_overfitting(D_real, D_fake): # input shape: (N, 1, H, W)
    real_logit = D_real.mean()
    fake_logit = D_fake.mean()
    return real_logit, fake_logit

# save and load
def save_model(models, epoch, iter, parallel, path):
    if isinstance(models, list):
        for model in models:
            if parallel:
                name = 'epoch' + str(epoch) + 'iter' + str(iter) + model.module.__class__.__name__
                save_path = os.path.join(path, name)
                torch.save(model.module.state_dict(), save_path)
            else:
                name = 'epoch' + str(epoch) + 'iter' + str(iter) + model.__class__.__name__
                save_path = os.path.join(path, name)
                torch.save(model.state_dict(), save_path)
    else:
        model = models
        if parallel:
            name = 'epoch' + str(epoch) + 'iter' + str(iter) + model.module.__class__.__name__
            save_path = os.path.join(path, name)
            torch.save(model.module.state_dict(), save_path)
        else:
            name = 'epoch' + str(epoch) + 'iter' + str(iter) + model.__class__.__name__
            save_path = os.path.join(path, name)
            torch.save(model.state_dict(), save_path)


def load_model(model, path):
    load_path = path
    model.load_state_dict(torch.load(load_path))





    













