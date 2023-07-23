import os
import sys
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))) )
from PIL import Image
import random
import shutil

import torchvision.transforms as transforms

from core.setting import config

def preprocessing(file_path, save_root, data_shape, option=None, crop_size=None): # image load , crop, resize and save 
    img = Image.open(file_path)
    width, height = img.size
    
    if option == 'center_crop':
        if width > height:
            crop = transforms.CenterCrop(height)
        else:
            crop = transforms.CenterCrop(width)
        img = crop(img)
    elif option == 'crop':
        crop = transforms.CenterCrop((crop_size[0], crop_size[1]))
        img = crop(img)
    elif option == 'default':
        pass
    else:   
        print('please select option')

    img = img.resize(data_shape)
    img.save(os.path.join(save_root, os.path.basename(file_path)))
    # os.remove(file_path)

if __name__ == '__main__':
    cfg = config()

    data_dir = 'D:/DATA_ARCHIVE/DATASET/celeba_dataset'
    data_shape = (64, 64)
    new_data_root = cfg.dataset_root
    data_name = 'CelebA_teest'

    data_num_list = [100]

    data_list = []
    for (root, dirs, files) in os.walk(data_dir):
        if len(files) is not 0:
            for file in files:
                data_list.append(os.path.join(root, file))
    
    for data_num in data_num_list:
        save_root = os.path.join(new_data_root, data_name + '_' + str(data_num))

        if os.path.exists(save_root):
            shutil.rmtree(save_root)
        os.makedirs(save_root)

        sampled_files = random.sample(data_list, data_num)
        for sampled_file in sampled_files:
            preprocessing(sampled_file, save_root, data_shape=data_shape, option='center_crop', crop_size=data_shape)