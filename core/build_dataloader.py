from PIL import Image
import os
import numpy as np
import cv2 as cv

import torch
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision

from core.func import *

def preprocessing(data_root, option=None):
    preprocessed_data = []

    if option == 'load_path':
        for (root, dirs, files) in os.walk(data_root):
            for file in files:
                preprocessed_data.append(os.path.join(root, file))
    elif option == 'load_direct':
        for (root, dirs, files) in os.walk(data_root):
            for file in files:
                file = os.path.join(root, file)
                for func in compose_transform(option='load_path'):
                    file = func(file)
                preprocessed_data.append(file)
    else:
        print('please select option')
        return
    
    return preprocessed_data

def img_resize(input, shape=(64,64)):
    return torchvision.transforms.functional.resize(input, shape)

def compose_transform(option=None):
    if option == 'load_path':
        return [open_image, numpy2tensor, img_resize, minmax_scale]
    elif option == 'load_direct':
        return None


class my_transform (object):
    def __init__(self, process_list):
        self.process_list = process_list
    
    def __call__(self, data):
        if self.process_list == None:
            return data
        else:
            for process in self.process_list:
                data = process(data)
            return data


class SimpleDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.transform.process_list == None:
            data = self.data_list[idx]
        else:
            data = self.transform(self.data_list[idx])

        return data