import os
import sys

from torch.nn.modules import padding
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.conv import ConvTranspose2d
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))) )

from collections import namedtuple

import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.models as models

from core.setting import config

cfg = config()

def inverse_attention(feat_map, aux_cls, aug_task):
    # just for single label
    # aug task shape: [N, C]
    # feat_map shape: [N, C, H, W]
    aug_weight = torch.cat([aux_cls.linear.weight[task:task+1] for task in aug_task], dim=0) # shape (N, C)
    aug_weight = torch.unsqueeze(torch.unsqueeze(torch.sigmoid(-aug_weight), dim=2), dim=3)
    att_feat_map = feat_map * aug_weight

    return att_feat_map

    

class Skip_Generator(nn.Module): # Skip Generator
    def __init__(self, latent_channel=cfg.latent_shape[0], d=128, norm=False, spectral_norm=True, upsample_method='interpolate'):
        super().__init__()
        self.init_channel = latent_channel

        self.init_blk = Conv_Blk_From_z(self.init_channel, d*8, norm=norm, spectral_norm=spectral_norm)
        self.skip_blk1 = Skip_Blk(d*8, d*4, norm=norm, spectral_norm=spectral_norm, upsample_method=upsample_method)
        self.skip_blk2 = Skip_Blk(d*4, d*2, norm=norm, spectral_norm=spectral_norm, upsample_method=upsample_method)
        self.skip_blk3 = Skip_Blk(d*2, d*1, norm=norm, spectral_norm=spectral_norm, upsample_method=upsample_method)
        self.skip_blk4 = Skip_Blk(d, 3, norm=norm, spectral_norm=spectral_norm, upsample_method=upsample_method)
    
    def forward(self, input):
        x = self.init_blk(input) # x shape is (N, C, 4, 4)
        x, rgb_1 = self.skip_blk1(x)
        rgb = F.interpolate(rgb_1, scale_factor=2, mode='nearest')
        x, rgb_2 = self.skip_blk2(x)
        rgb = F.interpolate(rgb + rgb_2, scale_factor=2, mode='nearest')
        x, rgb_3 = self.skip_blk3(x)
        rgb = F.interpolate(rgb + rgb_3, scale_factor=2, mode='nearest')
        x, rgb_4 = self.skip_blk4(x)
        rgb = torch.tanh(rgb + rgb_4)
        return rgb, rgb_1.detach(), rgb_2.detach(), rgb_3.detach(), rgb_4.detach()

class Res_Generator(nn.Module): # Res Generator
    def __init__(self, latent_channel=cfg.latent_shape[0], d=128, norm='BN', spectral_norm=True):
        super().__init__()
        self.init_channel = latent_channel

        self.init_blk = Conv_Blk_From_z(self.init_channel, d*8, norm=norm, spectral_norm=spectral_norm)
        self.res_blk1 = Res_Blk(d*8, d*4, norm=norm, spectral_norm=spectral_norm, sampling='upsample')
        self.res_blk2 = Res_Blk(d*4, d*2, norm=norm, spectral_norm=spectral_norm, sampling='upsample')
        self.res_blk3 = Res_Blk(d*2, d*1, norm=norm, spectral_norm=spectral_norm, sampling='upsample')
        self.res_blk4 = Res_Blk(d*1, 3, norm=norm, spectral_norm=spectral_norm, sampling='upsample')

    def forward(self, input):
        x = self.init_blk(input)
        x = self.res_blk1(x)
        x = self.res_blk2(x)
        x = self.res_blk3(x)
        out = torch.tanh(self.res_blk4(x))
        return out


class DC_Generator(nn.Module): # DCGAN Generator
    # initializers
    def __init__(self, d=128):
        super().__init__()
        
        self.deconv1 = nn.ConvTranspose2d(128, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
        
        """
        self.deconv1 = nn.ConvTranspose2d(128, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.conv2 = nn.Conv2d(d*8, d*4, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(d*4)
        self.conv3 = nn.Conv2d(d*4, d*2, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(d*2)
        self.conv4 = nn.Conv2d(d*2, d, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d)
        self.conv5 = nn.Conv2d(d, 3, 3, 1, 1)
        
        self.actv = nn.LeakyReLU(0.2)
        """

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        """
        x = self.actv(self.deconv1_bn(self.deconv1(input)))
        x = self.actv(self.conv2_bn(self.conv2(F.interpolate(x, scale_factor=2, mode='nearest'))))
        x = self.actv(self.conv3_bn(self.conv3(F.interpolate(x, scale_factor=2, mode='nearest'))))
        x = self.actv(self.conv4_bn(self.conv4(F.interpolate(x, scale_factor=2, mode='nearest'))))
        x = torch.tanh(self.conv5(F.interpolate(x, scale_factor=2, mode='nearest')))
        """
        return x

class Res_Discriminator(nn.Module): # Res Discriminator
    def __init__(self, norm='BN', spectral_norm=True, d=128):
        super().__init__()
        from core.train import cfg
        self.spectral_norm = spectral_norm

        self.from_rgb = nn.Sequential(self._spectral_norm(nn.Conv2d(3, d, 1)), nn.LeakyReLU(0.2))
        self.res_blk1 = Res_Blk(d, d, norm=norm, spectral_norm=spectral_norm)
        self.res_blk2 = Res_Blk(d, d*2, norm=norm, spectral_norm=spectral_norm)
        self.res_blk3 = Res_Blk(d*2, d*4, norm=norm, spectral_norm=spectral_norm)
        self.res_blk4 = Res_Blk(d*4, d*8, norm=norm, spectral_norm=spectral_norm)
        self.final_conv = self._spectral_norm(nn.Conv2d(d*8, 1, 4))
        self.aux_cls = Aux_cls(d*8, cfg.aug_num, spectral_norm=spectral_norm)
    
    def _spectral_norm(self, layer):
        from torch.nn.utils import spectral_norm as sn
        if self.spectral_norm:
            return sn(layer)
        else:
            return layer
    
    def forward(self, input):
        x = self.from_rgb(input)
        x = self.res_blk1(x)
        x = self.res_blk2(x)
        x = self.res_blk3(x)
        x = self.res_blk4(x)
        aux_out = self.aux_cls(x)
        out = self.final_conv(x)
        return out, aux_out

class DC_Discriminator(nn.Module): # DCGAN Discriminator
    # initializers
    def __init__(self, d=128):
        from core.train import cfg
        super().__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
        self.aux_cls = Aux_cls(d*8, cfg.aug_num)
        """
        self.conv1 = nn.Conv2d(3, d, 3, 1, 1)
        self.conv1_bn = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, d*2, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
        self.aux_cls = Aux_cls(d*8, cfg.aug_num)

        self.actv = nn.LeakyReLU(0.2)
        """

    # forward method
    def forward(self, input, aug_task):

        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        aux_out = self.aux_cls(x)
        if cfg.use_attention:
            x = inverse_attention(x, self.aux_cls, aug_task)
        out = self.conv5(x)

        """
        x = F.avg_pool2d(self.actv(self.conv1(input)), 2)
        x = F.avg_pool2d(self.actv(self.conv2_bn(self.conv2(x))), 2)
        x = F.avg_pool2d(self.actv(self.conv3_bn(self.conv3(x))), 2)
        x = F.avg_pool2d(self.actv(self.conv4_bn(self.conv4(x))), 2)
        aux_out = self.aux_cls(x)
        if cfg.use_attention:
            x = inverse_attention(x, self.aux_cls, aug_task)
        out = self.conv5(x)
        """
        return out, aux_out

class Conv_Blk_From_z(nn.Module):
    def __init__(self, in_channels, out_channels, norm='BN', spectral_norm=True):
        super().__init__()
        self.spectral_norm = spectral_norm
        self.norm = norm

        self.conv1 = self.conv_from_z(in_channels, out_channels)
        self.conv2 = self.conv_blk(out_channels, out_channels)
    
    def conv_from_z(self, in_channels, out_channels):
        layer_list = []

        conv = self._spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, 4, 1))
        norm = self._choose_norm(out_channels)
        actv = nn.LeakyReLU(0.2)

        layer_list.append(conv)
        if norm is not None:
            layer_list.append(norm)
        layer_list.append(actv)

        return nn.Sequential(*layer_list)
    
    def conv_blk(self, in_channels, out_channels):
        layer_list = []

        conv = self._spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        norm = self._choose_norm(out_channels)
        actv = nn.LeakyReLU(0.2)

        layer_list.append(conv)
        if norm is not None:
            layer_list.append(norm)
        layer_list.append(actv)
        
        return nn.Sequential(*layer_list)

    def _choose_norm(self, channels):
        if self.norm == 'BN':
            return nn.BatchNorm2d(channels)
        elif self.norm == 'IN':
            return nn.InstanceNorm2d(channels)
        elif self.norm == False:
            return None

    def _spectral_norm(self, layer):
        from torch.nn.utils import spectral_norm as sn
        if self.spectral_norm:
            return sn(layer)
        else:
            return layer
    
    def forward(self, input):
        x = self.conv1(input)
        out = self.conv2(x)
        return out

class Skip_Blk(nn.Module):
    def __init__(self, in_channels, out_channels, norm='BN', spectral_norm=True, upsample_method='interpolate'):
        """
        upsample_method: option1 = 'interpolate, option2 = 'transposeconv'
        """
        super().__init__()
        self.spectral_norm = spectral_norm
        self.norm = norm
        self.upsample_method = upsample_method

        self.conv = self.conv_blk(in_channels, out_channels)
        self.to_rgb = self._spectral_norm(nn.Conv2d(out_channels, 3, 1))
    
    def conv_blk(self, in_channels, out_channels):
        layer_list = []

        if self.upsample_method == 'interpolate':
            conv1 = self._spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        elif self.upsample_method == 'transposeconv':
            conv1 = self._spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
        norm1 = self._choose_norm(out_channels)
        actv1 = nn.LeakyReLU(0.2)

        conv2 = self._spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        norm2 = self._choose_norm(out_channels)
        actv2 = nn.LeakyReLU(0.2)

        if self.upsample_method == 'interpolate':
            layer_list.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layer_list.append(conv1)
        if norm1 is not None:
            layer_list.append(norm1)
        layer_list.append(actv1)
        layer_list.append(conv2)
        if norm2 is not None:
            layer_list.append(norm2)
        layer_list.append(actv2)

        return nn.Sequential(*layer_list)

    def _choose_norm(self, channels):
        if self.norm == 'BN':
            return nn.BatchNorm2d(channels)
        elif self.norm == 'IN':
            return nn.InstanceNorm2d(channels)
        elif self.norm == False:
            return None

    def _spectral_norm(self, layer):
        from torch.nn.utils import spectral_norm as sn
        if self.spectral_norm:
            return sn(layer)
        else:
            return layer

    def forward(self, input):
        out = self.conv(input)
        rgb = self.to_rgb(out)
        return out, rgb

class Res_Blk(nn.Module):
    def __init__(self, in_channels, out_channels, norm='BN', spectral_norm=True, sampling='downsample'):
        super().__init__()
        self.spectral_norm = spectral_norm
        self.sampling = sampling
        self.norm = norm

        self.conv_shortcut = self.shortcut_blk(in_channels, out_channels)
        self.res_path1 = self.res_blk(in_channels, in_channels)
        self.res_path2 = self.res_blk(in_channels, out_channels)
    
    def shortcut_blk(self, in_channels, out_channels):
        conv1x1 = self._spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
        return conv1x1

    def res_blk(self, in_channels, out_channels):
        layer_list = []

        norm = self._choose_norm(in_channels)
        actv = nn.LeakyReLU(0.2)
        conv = self._spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1))

        if norm is not None:
            layer_list.append(norm)
        layer_list.append(actv)
        layer_list.append(conv)

        return nn.Sequential(*layer_list)
        
    def _choose_norm(self, channels):
        if self.norm == 'BN':
            return nn.BatchNorm2d(channels)
        elif self.norm == 'IN':
            return nn.InstanceNorm2d(channels)
        elif self.norm == False:
            return None

    def _spectral_norm(self, layer):
        from torch.nn.utils import spectral_norm as sn
        if self.spectral_norm:
            return sn(layer)
        else:
            return layer
    
    def _sampling(self, input):
        if self.sampling == 'upsample':
            return F.interpolate(input, scale_factor=2, mode='nearest')
        elif self.sampling == 'downsample':
            return F.avg_pool2d(input, 2)
    
    def _shortcut(self, input):
        x = self.conv_shortcut(input)
        out = self._sampling(x)
        return out

    def _residual(self, input):
        x = self.res_path1(input)
        x = self._sampling(x)
        out = self.res_path2(x)
        return out
    
    def forward(self, input):
        out = self._shortcut(input) + self._residual(input)
        return out

class Aux_cls(nn.Module):
    def __init__(self, channel_num, aug_num, spectral_norm=False):
        super().__init__()
        self.spectral_norm = spectral_norm

        self.linear = nn.Linear(channel_num, aug_num, bias=False)
        self.channel_num = channel_num

    def _spectral_norm(self, layer):
        from torch.nn.utils import spectral_norm as sn
        if self.spectral_norm:
            return sn(layer)
        else:
            return layer
    
    def forward(self, input):
        pooling_kernel_size = input.shape[2:4]
        gap = torch.squeeze(F.avg_pool2d(input, kernel_size=pooling_kernel_size))
        out = self.linear(gap)
        return out

class VGG(nn.Module):
    def __init__(self, requires_grad=True):
        super(VGG, self).__init__()
        vgg_pretrained_features = models.vgg19_bn(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        """
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        """
        for x in range(6):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 10):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        """
        for x in range(13, 26):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(26, 39):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        """
        # additional
        #self.slice1.to(device)
        #self.slice2.to(device)

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        """
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        """
        out = [h_relu1_2, h_relu2_2]
        return out