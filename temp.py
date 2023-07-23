from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2 as cv
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
from varname import nameof
from core.setting import config

"""
img = cv.imread('./000177.jpg') # BGR, (H, W, C)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float64) # RGB, (H, W, C)
#img = np.transpose(img, (2, 0, 1)) # RGB (C, H, W)
"""
"""
img = Image.open('./000177.jpg')
#img = np.array(img)

#ts = transforms.GaussianBlur(kernel_size=3)
ts = transforms
img2 = ts(img)

grad = torch.autograd.grad(outputs=out.sum(), inputs=img, create_graph=True, only_inputs=True)

c = 3
"""

layer = nn.Linear(100,20)

x = 3
