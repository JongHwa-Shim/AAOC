import os
import sys

from torch.nn.modules import padding
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

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_channels, affine=False)
        self.affine = nn.Linear(style_dim, num_channels*2) 

    def forward(self, x, style):
        style_vec = self.affine(style) # output (N, num_channels*2)
        style_vec = style_vec.view(style_vec.size(0), style_vec.size(1), 1, 1) # output (N, num_channels*2, 1, 1)
        gamma, beta = torch.chunk(style_vec, chunks=2, dim=1) # gamma, beta (N, num_channels, 1, 1)
        
        return (1 + gamma) * self.norm(x) + beta # gamma is additional value for 1

class Noise_layer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(num_channels))
        self.noise = None
    
    def forward(self, input):
        if self.noise is None:
            noise = torch.randn(input.shape[0], 1, input.shape[2], input.shape[3], device=input.device, dtype=input.dtype)
        else:
            noise = self.noise

        out = input + self.weight.view(1, -1, 1, 1) * noise
        return out

class Style_encoder(nn.Module):
    def __init__(self, input_dim=cfg.latent_shape, output_dim=cfg.latent_shape, hidden_dim=512, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            actv,
            nn.Linear(hidden_dim, hidden_dim),
            actv,
            nn.Linear(hidden_dim, hidden_dim),
            actv,
            nn.Linear(hidden_dim, hidden_dim),
            actv,
            nn.Linear(hidden_dim, hidden_dim),
            actv,
            nn.Linear(hidden_dim, hidden_dim),
            actv,
            nn.Linear(hidden_dim, hidden_dim),
            actv,
            nn.Linear(hidden_dim, output_dim),
            actv
        )
    
    def forward(self, input):
        return self.encoder(input)

class Conv_Block(nn.Module):
    def __init__(self, init_channel, out_channel, resample=None, actv='ReLU', normalize='AdaIN', use_noise=False):
        super().__init__()
        if resample == 'downsample':
            self.resample_ratio = 0.5
        elif resample == 'upsample':
            self.resample_ratio = 2
        else:
            raise(ValueError)
        self.resample = resample

        module_list_1 = []
        module_list_1.append(nn.ReflectionPad2d(1))
        module_list_1.append(nn.Conv2d(init_channel, init_channel, 3, 1, 0))
        if use_noise:
            module_list_1.append(Noise_layer(init_channel))
        module_list_1.append(nn.ReLU())
        self.block1 = nn.Sequential(*module_list_1)

        self.norm1 = AdaIN(cfg.latent_shape, init_channel)

        module_list_2 = []
        module_list_2.append(nn.ReflectionPad2d(1))
        module_list_2.append(nn.Conv2d(init_channel, out_channel, 3, 1, 0))
        if use_noise:
            module_list_1.append(Noise_layer(out_channel))
        module_list_2.append(nn.ReLU())
        self.block2 = nn.Sequential(*module_list_2)

        self.norm2 = AdaIN(cfg.latent_shape, out_channel)
        
    def forward(self, input):
        x = input[0]
        style = input[1]

        if self.resample == 'downsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')
        elif self.resample == 'upsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')

        x = self.block1(x)
        x = self.norm1(x, style)
        x = self.block2(x)
        x = self.norm2(x, style)

        return [x, style]

class Res_Block(nn.Module):
    def __init__(self, init_channel, actv='ReLU', normalize='AdaIN', use_noise=False):
        super().__init__()  
        module_list_1 = []
        module_list_1.append(nn.ReflectionPad2d(1))
        module_list_1.append(nn.Conv2d(init_channel, init_channel, 3, 1, 0))
        if use_noise:
            module_list_1.append(Noise_layer(init_channel))
        module_list_1.append(nn.ReLU())
        self.block1 = nn.Sequential(*module_list_1)

        self.norm1 = AdaIN(cfg.latent_shape, init_channel)

        module_list_2 = []
        module_list_2.append(nn.ReflectionPad2d(1))
        module_list_2.append(nn.Conv2d(init_channel, init_channel, 3, 1, 0))
        if use_noise:
            module_list_1.append(Noise_layer(init_channel))
        module_list_2.append(nn.ReLU())
        self.block2 = nn.Sequential(*module_list_2)

        self.norm2 = AdaIN(cfg.latent_shape, init_channel)
        
    def forward(self, input):
        x_ = input[0]
        style = input[1]

        x = self.block1(x_)
        x = self.norm1(x, style)
        x = self.block2(x)
        x = self.norm2(x, style)

        return [x_ + x, style]

class Dis_Block(nn.Module):
    def __init__(self, init_channel, out_channel, actv='LeakyReLU'):
        super().__init__()
        module_list = [nn.ReflectionPad2d(1), nn.utils.spectral_norm(nn.Conv2d(init_channel, init_channel, 4, 2, 0)),
                nn.LeakyReLU(0.2)]
        
        module_list += [nn.ReflectionPad2d(1), nn.utils.spectral_norm(nn.Conv2d(init_channel, out_channel, 3, 1, 0)),
                nn.LeakyReLU(0.2)]

        self.block = nn.Sequential(*module_list)

    def forward(self, input):
        x = self.block(input)
        return x

class Dis_Block_end(nn.Module):
    def __init__(self, init_channel, out_channel, actv='LeakyReLU'):
        super().__init__()
        module_list = [nn.ReflectionPad2d(1), nn.utils.spectral_norm(nn.Conv2d(init_channel, init_channel, 3, 1, 0)),
                nn.LeakyReLU(0.2)]
        
        module_list += [nn.ReflectionPad2d(1), nn.utils.spectral_norm(nn.Conv2d(init_channel, out_channel, 3, 1, 0)),
                nn.LeakyReLU(0.2)]

        self.block = nn.Sequential(*module_list)

    def forward(self, input):
        x = self.block(input)
        return x

class _ResBlk(nn.Module):
    def __init__(self, init_channel, out_channel, actv=nn.LeakyReLU(0.2), normalize='AdaIN', resample=None, use_noise=False):
        super().__init__()
        # stargan v2 version
        self.actv = actv
        self.normalize = normalize
        self.resample = resample
        self.channel_change = init_channel != out_channel
        self.use_noise = use_noise
        self._build_weights(init_channel, out_channel)
        
        if self.resample == 'downsample':
            self.resample_ratio = 0.5
        elif self.resample == 'upsample':
            self.resample_ratio = 2
    
    class Noise_layer(nn.Module):
        def __init__(self, num_channels):
            super().__init__()
            self.weight = nn.Parameter(torch.rand(num_channels))
            self.noise = None
        
        def forward(self, input, noise=None):
            if noise is None and self.noise is None:
                noise = torch.randn(input.shape[0], 1, input.shape[2], input.shape[3], device=input.device, dtype=input.dtype)
            elif noise is None:
                noise = self.noise

            out = input + self.weight.view(1, -1, 1, 1) * noise
            return out

    def _no_normalize(self, *input):
        return [input[0]]
    
    def _in(self, *input):
        return [input[0]]
    
    def _adain(self, *input):
        return [input[0], input[1]]
    
    def _bypass(self, input):
        return input
    
    def _build_weights(self, init_channel, out_channel):
        self.conv1 = nn.Conv2d(init_channel, init_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(init_channel, out_channel, 3, 1, 1)
        
        if self.normalize == 'IN':
            self.norm1 = nn.InstanceNorm2d(init_channel, affine=True)
            self.norm2 = nn.InstanceNorm2d(init_channel, affine=True)
            self.norm_func = self._in
        elif self.normalize == 'AdaIN':
            self.norm1 = AdaIN(cfg.latent_shape, init_channel)
            self.norm2 = AdaIN(cfg.latent_shape, init_channel)
            self.norm_func = self._adain
        else:
            self.norm1 = self._bypass
            self.norm2 = self._bypass
            self.norm_func = self._no_normalize

        if self.channel_change:
            self.conv1x1 = nn.Conv2d(init_channel, out_channel, 1, 1, padding=0)
        
        if self.use_noise:
            self.noise1 = self.Noise_layer(init_channel)
            self.noise2 = self.Noise_layer(out_channel)
        
    def _shortcut(self, x):
        if self.channel_change:
            x = self.conv1x1(x)

        if self.resample == 'downsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')
        elif self.resample == 'upsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')

        return x

    def _residual(self, x, style):
        x = self.norm1(*self.norm_func(x, style))

        x = self.actv(x)
        x = self.conv1(x)

        if self.use_noise:
            x = self.noise1(x)

        if self.resample == 'downsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')
        elif self.resample == 'upsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')
        
        x = self.norm2(*self.norm_func(x, style))

        x = self.actv(x)
        x = self.conv2(x)

        if self.use_noise:
            x = self.noise2(x)

        return x

    def forward(self, input, style=None):
        output = self._shortcut(input) + self._residual(input, style)
        if self.normalize == 'AdaIN':
            return output, style
        else: return output

class ResBlk(nn.Module):
    def __init__(self, init_channel, out_channel, actv=nn.LeakyReLU(0.2), normalize='AdaIN', resample=None, use_noise=False):
        super().__init__()
        # stargan v2 version
        self.actv = actv
        self.normalize = normalize
        self.resample = resample
        self.channel_change = init_channel != out_channel
        self.use_noise = use_noise
        self._build_weights(init_channel, out_channel)
        
        if self.resample == 'downsample':
            self.resample_ratio = 0.5
        elif self.resample == 'upsample':
            self.resample_ratio = 2
    
    class Noise_layer(nn.Module):
        def __init__(self, num_channels):
            super().__init__()
            self.weight = nn.Parameter(torch.rand(num_channels))
            self.noise = None
        
        def forward(self, input, noise=None):
            if noise is None and self.noise is None:
                noise = torch.randn(input.shape[0], 1, input.shape[2], input.shape[3], device=input.device, dtype=input.dtype)
            elif noise is None:
                noise = self.noise

            out = input + self.weight.view(1, -1, 1, 1) * noise
            return out

    def _no_normalize(self, *input):
        return [input[0]]
    
    def _in(self, *input):
        return [input[0]]
    
    def _adain(self, *input):
        return [input[0], input[1]]
    
    def _bypass(self, input):
        return input
    
    def _build_weights(self, init_channel, out_channel):
        self.conv1 = nn.Conv2d(init_channel, init_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(init_channel, out_channel, 3, 1, 1)
        
        if self.normalize == 'IN':
            self.norm1 = nn.InstanceNorm2d(init_channel, affine=True)
            self.norm2 = nn.InstanceNorm2d(init_channel, affine=True)
            self.norm_func = self._in
        elif self.normalize == 'AdaIN':
            self.norm1 = AdaIN(cfg.latent_shape, init_channel)
            self.norm2 = AdaIN(cfg.latent_shape, init_channel)
            self.norm_func = self._adain
        else:
            raise(ValueError)

        if self.channel_change:
            self.conv1x1 = nn.Conv2d(init_channel, out_channel, 1, 1, padding=0)
        
        if self.use_noise:
            self.noise1 = self.Noise_layer(init_channel)
            self.noise2 = self.Noise_layer(out_channel)
        
    def _shortcut(self, x):
        if self.channel_change:
            x = self.conv1x1(x)

        if self.resample == 'downsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')
        elif self.resample == 'upsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')

        return x

    def _residual(self, x, style):
        x = self.norm1(*self.norm_func(x, style))

        x = self.actv(x)
        x = self.conv1(x)

        if self.use_noise:
            x = self.noise1(x)

        if self.resample == 'downsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')
        elif self.resample == 'upsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')
        
        x = self.norm2(*self.norm_func(x, style))

        x = self.actv(x)
        x = self.conv2(x)

        if self.use_noise:
            x = self.noise2(x)

        return x

    def forward(self, input, style=None):
        output = self._shortcut(input) + self._residual(input, style)
        if self.normalize == 'AdaIN':
            return output, style
        else: return output

class ResBlk_G(nn.Module):
    def __init__(self, init_channel, out_channel, normalize='AdaIN', resample=None, use_noise=False, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        # stargan v2 version
        self.actv = actv
        self.normalize = normalize
        self.resample = resample
        self.channel_change = init_channel != out_channel
        self.use_noise = use_noise
        self._build_weights(init_channel, out_channel)
        
        if self.resample == 'downsample':
            self.resample_ratio = 0.5
        elif self.resample == 'upsample':
            self.resample_ratio = 2
    
    class Noise_layer(nn.Module):
        def __init__(self, num_channels):
            super().__init__()
            self.weight = nn.Parameter(torch.rand(num_channels))
            self.noise = None
        
        def forward(self, input, noise=None):
            if noise is None and self.noise is None:
                noise = torch.randn(input.shape[0], 1, input.shape[2], input.shape[3], device=input.device, dtype=input.dtype)
            elif noise is None:
                noise = self.noise

            out = input + self.weight.view(1, -1, 1, 1) * noise
            return out
    
    def _build_weights(self, init_channel, out_channel):
        self.conv1 = nn.Conv2d(init_channel, init_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(init_channel, out_channel, 3, 1, 1)
        
        if self.normalize == 'IN':
            self.norm1 = nn.InstanceNorm2d(init_channel, affine=True)
            self.norm2 = nn.InstanceNorm2d(init_channel, affine=True)
        elif self.normalize == 'AdaIN':
            self.norm1 = AdaIN(cfg.latent_shape, init_channel)
            self.norm2 = AdaIN(cfg.latent_shape, init_channel)
        else:
            raise(ValueError)

        if self.channel_change:
            self.conv1x1 = nn.Conv2d(init_channel, out_channel, 1, 1, padding=0)
        
        if self.use_noise:
            self.noise1 = self.Noise_layer(init_channel)
            self.noise2 = self.Noise_layer(out_channel)
        
    def _shortcut(self, x):
        if self.channel_change:
            x = self.conv1x1(x)

        if self.resample == 'downsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')
        elif self.resample == 'upsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')

        return x

    def _residual(self, x, style):
        if self.normalize == 'AdaIN':
            x = self.norm1(x, style)
        else:
            x = self.norm1(x)

        x = self.actv(x)
        x = self.conv1(x)

        if self.use_noise:
            x = self.noise1(x)

        if self.resample == 'downsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')
        elif self.resample == 'upsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')
        
        if self.normalize == 'AdaIN':
            x = self.norm2(x, style)
        else:
            x = self.norm2(x)

        x = self.actv(x)
        x = self.conv2(x)

        if self.use_noise:
            x = self.noise2(x)

        return x

    def forward(self, input, style=None):
        output = self._shortcut(input) + self._residual(input, style)

        if self.normalize == 'AdaIN':
            return output, style
        else: 
            return output

class ResBlk_D(nn.Module):
    def __init__(self, init_channel, out_channel, resample=None, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        # stargan v2 version
        self.actv = actv
        self.resample = resample
        self.channel_change = init_channel != out_channel
        self._build_weights(init_channel, out_channel)
        
        if self.resample == 'downsample':
            self.resample_ratio = 0.5
        elif self.resample == 'upsample':
            self.resample_ratio = 2
    
    def _build_weights(self, init_channel, out_channel):
        self.conv1 = nn.Conv2d(init_channel, init_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(init_channel, out_channel, 3, 1, 1)

        if self.channel_change:
            self.conv1x1 = nn.Conv2d(init_channel, out_channel, 1, 1, padding=0)
        
    def _shortcut(self, x):
        if self.channel_change:
            x = self.conv1x1(x)

        if self.resample == 'downsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')
        elif self.resample == 'upsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')

        return x

    def _residual(self, x):
        x = self.actv(x)
        x = self.conv1(x)

        if self.resample == 'downsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')
        elif self.resample == 'upsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='bilinear')

        x = self.actv(x)
        x = self.conv2(x)

        return x

    def forward(self, input):
        output = self._shortcut(input) + self._residual(input)

        return output

class Aux_cls(nn.Module):
    def __init__(self, aug_num, channel_num):
        super().__init__()
        self.linear = nn.Linear(channel_num, aug_num)
        self.channel_num = channel_num

    def attention(self, input, aug_label=None):
        None
    
    def forward(self, input):
        pooling_kernel_size = input.shape[2:4]
        gap = torch.squeeze(F.avg_pool2d(input, kernel_size=pooling_kernel_size))
        out = self.linear(gap)
        return out


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x

class Discriminator(nn.Module):
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
        #self.aux_cls = Aux_cls(cfg.aug_num, d*8)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        #aux_out = self.aux_cls(x)
        x = self.conv5(x)

        return x, None

class _Generator(nn.Module): # name: default_g
    def __init__(self):
        super().__init__()
        # input size = (N, 64, 4, 4)
        # output size = (3, 256, 256)
        self.style_encoder = Style_encoder()

        self.bottle_neck_1 = mySequential(
            ResBlk_G(64, 128, normalize='AdaIN', resample='upsample'), # 8x8
            ResBlk_G(128, 128, normalize='AdaIN', resample='upsample') # 16x16
        )
        
        self.encoder_1 = mySequential(
            ResBlk_G(128, 256, normalize='AdaIN', resample='upsample'), # 32x32
            ResBlk_G(256, 512, normalize='AdaIN', resample='upsample') #64x 64
        )
        
        self.bottle_neck_2 = mySequential(
            ResBlk_G(512, 512, normalize='AdaIN'),
            ResBlk_G(512, 256, normalize='AdaIN')
        )

        self.encoder_2 = mySequential(
            ResBlk_G(256, 128, normalize='AdaIN', resample='upsample'), # 128x128
            ResBlk_G(128, 64, normalize='AdaIN', resample='upsample') # 256x256
        )

        self.layer = mySequential(
            ResBlk_G(8, 16, normalize='AdaIN', resample='upsample'),
            ResBlk_G(16, 32, normalize='AdaIN', resample='upsample'),
            ResBlk_G(32, 64, normalize='AdaIN', resample='upsample'),
            ResBlk_G(64, 3, normalize='AdaIN')
        )

        self.to_rgb = ResBlk_G(64, 3, normalize='AdaIN')
        self.scaling = nn.Tanh()

    def forward(self, input, style): # input is always (512, H/16, W/16)
        style_vec = self.style_encoder(style)
        """x, _ = self.bottle_neck_1(input, style_vec)
        x, _ = self.encoder_1(x, style_vec)
        x, _ = self.bottle_neck_2(x, style_vec)
        x, _ = self.encoder_2(x, style_vec)
        x, _ = self.to_rgb(x, style_vec)"""
        x, _ = self.layer(input, style_vec)
        x = self.scaling(x)
        return x


        return x # outputsize = (3, H, W)

class _Discriminator(nn.Module): # name: default_discriminator
    def __init__(self):
        super().__init__()
        # input size: (3, H, W)
        self.from_rgb_conv1x1 = ResBlk_D(3, 64) # output size (64, H, W)

        self.encoder = nn.Sequential(
            ResBlk_D(64, 128, resample='downsample'), # output size (128, H/2, W/2)
            ResBlk_D(128, 256, resample='downsample'), # output size (256, H/4, W/4)
            ResBlk_D(256, 512, resample='downsample')#, # output size (512, H/8, W/8)
            #ResBlk_D(512, 1024, resample='downsample'), # output size (1024, H/16, W/16)
            #ResBlk_D(1024, 2048) # output size (2048, H/16, W/16)
        )

        self.gap_aux_cls = (nn.Linear(512, cfg.aug_num))
        self.gmp_aux_cls = (nn.Linear(512, cfg.aug_num))
        #self.gap_aux_cls = (nn.Linear(2048, cfg.aug_num))
        #self.gmp_aux_cls = (nn.Linear(2048, cfg.aug_num))

        self.conv1x1 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(1024, 512, kernel_size=1, stride=1))
        #self.conv1x1 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(4096, 2048, kernel_size=1, stride=1))
        self.cls = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(512, 1, kernel_size=1, stride=1))
        #self.cls = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(2048, 1, kernel_size=1, stride=1))

    def _aug_attention(self, x, aux_cls, aug_task):
        linear_weight = aux_cls.weight # (out, in) e.g. (10, 100)

        attention_list = [linear_weight[task:task+1] for task in aug_task]

        attention_mat = torch.cat(attention_list, dim=0) # (N, in)
        attention_mat.unsqueeze_(2).unsqueeze_(3) # (N, in, 1, 1)

        return x * attention_mat
            
        
    def forward(self, input, teacher_aug_task=None):
        x = self.from_rgb_conv1x1(input)
        x = self.encoder(x)
        
        # global average pooling attention
        pooling_kernel_size = x.shape[2:4]
        gap_x = torch.squeeze(F.avg_pool2d(x, kernel_size=pooling_kernel_size)) # (N, C)
        gap_logit = self.gap_aux_cls(gap_x) # (N, task)

        if teacher_aug_task is not None: # teacher force learning
            gap_aug_task = teacher_aug_task
        else: 
            gap_aug_task = torch.argmax(gap_logit, dim=1)
        
        gap_att_feature = self._aug_attention(x, self.gap_aux_cls, gap_aug_task)

        # global max pooling attention
        gmp_x = torch.squeeze(F.max_pool2d(x, kernel_size=pooling_kernel_size)) # (N, C)
        gmp_logit = self.gmp_aux_cls(gmp_x)

        if teacher_aug_task is not None: # teacher force learning
            gmp_aug_task = teacher_aug_task
        else:
            gmp_aug_task = torch.argmax(gmp_logit, dim=1)

        gmp_att_feature = self._aug_attention(x, self.gmp_aux_cls, gmp_aug_task)

        x = torch.cat([gap_att_feature, gmp_att_feature], 1)
        x = self.conv1x1(x)
        out = self.cls(x)

        heat_map = torch.sum(x, dim=1, keepdim=True).detach()

        return out, gap_logit, gmp_logit, heat_map

class Image_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # input size = (N, 3, H, W)
        self.from_rgb_conv1x1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1) # output size (N, 64, H/2, W/2)

        self.layer = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # output size (N, 128, H/4, W/4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # output size (N, 256, H/8, W/8)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # output size (N, 512, H/16, W/16)
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1) # output size (N, 1024, H/32, W/32)
        )

        self.gaussian_mapper = nn.Linear(1024, cfg.latent_shape)
    
    def forward(self, input):
        x = self.from_rgb_conv1x1(input)
        x = self.layer(x)
        x = F.avg_pool2d(x, kernel_size=x.shape[2:4]).squeeze()
        out = self.gaussian_mapper(x)

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








# deprecated model
#########################################################################################
class deprecated_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.style_encoder = Style_encoder()

        self.from_rgb = nn.Conv2d(3, 64, 1, 1, padding=0) # output size = (64, H, W)

        self.encode = nn.Sequential(
            ResBlk(64, 128, normalize='AdaIN', resample='downsample'), # output size = (128, H/2, W/2)
            ResBlk(128, 256, normalize='AdaIN', resample='downsample'), # output size = (256, H/4, W/4)
            ResBlk(256, 512, normalize='AdaIN', resample='downsample'), # output size = (512, H/8, W/8)
            ResBlk(512, 512, normalize='AdaIN', resample='downsample') # output size = (512, H/16, W/16)

        )

        self.bottle_neck = nn.Sequential(
            ResBlk(512, 512, normalize='AdaIN'), # output size = (512, H/16, W/16)
            ResBlk(512, 512, normalize='AdaIN'), # output size = (512, H/16, W/16)
            ResBlk(512, 512, normalize='AdaIN') # output size = (512, H/16, W/16)
        )

        self.decode = nn.Sequential(
        ResBlk(512, 256, normalize='AdaIN', resample='upsample'), # output size = (256, H/8, W/8)
        ResBlk(256, 128, normalize='AdaIN', resample='upsample'), # output size = (128, H/4, W/4)
        ResBlk(128, 64, normalize='AdaIN', resample='upsample'), # output size = (64, H/2, W/2)
        ResBlk(64, 32, normalize='AdaIN', resample='upsample') # output size = (32, H, W)
        )

        self.to_rgb = nn.Sequential(
            AdaIN(cfg.latent_shape, 32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 1, 1, 0),
            nn.Tanh()
        )    

    def forward(self, input, style):
        style_vec = self.style_encoder(style)

        x = self.from_rgb(input)
        x = self.encode(x, style_vec)
        x = self.bottle_neck(x, style_vec)
        x = self.decode(x, style_vec)

        x = self.to_rgb(x, style_vec)
        return x # outputsize = (3, H, W)

class __ResBlk(nn.Module):
    def __init__(self, init_channel, out_channel, actv=nn.LeakyReLU(0.2), normalize=False, resample=None, simplify=False):
        super().__init__()
        # stargan v2 version
        self.actv = actv
        self.normalize = normalize
        self.resample = resample
        self.channel_change = init_channel != out_channel
        self._build_weights(init_channel, out_channel)

        if simplify:
            self.resample_ratio = 4
        else:
            self.resample_ratio = 2

    def _build_weights(self, init_channel, out_channel):
        self.conv1 = nn.Conv2d(init_channel, init_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(init_channel, out_channel, 3, 1, 1)
        
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(init_channel, affine=True)
            self.norm2 = nn.InstanceNorm2d(init_channel, affine=True)

        if self.channel_change:
            self.conv1x1 = nn.Conv2d(init_channel, out_channel, 1, 1, padding=0)
        
    def _shortcut(self, x):
        if self.channel_change:
            x = self.conv1x1(x)

        if self.resample == 'downsample':
            x = F.avg_pool2d(x , self.resample_ratio)
        elif self.resample == 'upsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='nearest')

        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)

        x = self.actv(x)
        x = self.conv1(x)

        if self.resample == 'downsample':
            x = F.avg_pool2d(x, self.resample_ratio)
        elif self.resample == 'upsample':
            x = F.interpolate(x, scale_factor=self.resample_ratio, mode='nearest')
        
        if self.normalize:
            x = self.norm2(x)

        x = self.actv(x)
        x = self.conv2(x)

        return x

    def forward(self, input):
        output = self._shortcut(input) + self._residual(input)

        return output

"""class Generator(nn.Module): # input = (N, 64, 8, 8)
    def __init__(self):
        super().__init__()
        self.bottle_neck_1 = nn.Sequential(
            ResBlk(16, 64, normalize=True),
            ResBlk(64, 64, normalize=True),
            ResBlk(64, 64, normalize=True)
        )
        self.layer_1 = nn.Sequential(
            ResBlk(64, 128, normalize=True, resample='upsample'), # (128, 16, 16)
            ResBlk(128, 256, normalize=True, resample='upsample'), # (256, 32, 32)
            ResBlk(256, 512, normalize=True, resample='upsample') # (512, 64, 64)
        )

        self.bottle_neck_2 = nn.Sequential(
            ResBlk(512, 512, normalize=True),
            ResBlk(512, 512, normalize=True)
        )

        self.layer_2 = nn.Sequential(
            ResBlk(512, 256, normalize=True, resample='upsample'),
            ResBlk(256, 128, normalize=True),
            ResBlk(128, 64, normalize=True)
        )

        self.to_rgb = ResBlk(64, 3)
    
    def forward(self, input):
        x = self.bottle_neck_1(input)
        x = self.layer_1(x)
        x = self.bottle_neck_2(x)
        x = self.layer_2(x)
        x = self.to_rgb(x)
        x = F.tanh(x)
        return x"""