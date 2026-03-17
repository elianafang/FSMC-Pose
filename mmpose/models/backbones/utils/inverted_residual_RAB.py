# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule

from .se_layer import SELayer

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
import math
class RPReLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__() 
        self.move1 = nn.Parameter(torch.zeros(hidden_size))
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        out = self.prelu((x - self.move1).transpose(-1, -2)).transpose(-1, -2) + self.move2
        return out
    
class LearnableBiasnn(nn.Module): 
    def __init__(self, out_chn): 
        super(LearnableBiasnn, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, out_chn,1,1]), requires_grad=True)

    def forward(self, x): 
        out = x + self.bias.expand_as(x)
        return out

# Multi-scale grouped dilated convolution (MSGDC)    
class RAB(nn.Module):
    def __init__(self, in_chn, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1, padding='same'):
        super(RAB, self).__init__()
        self.move = LearnableBiasnn(in_chn)
        

        self.cov1 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation1, bias=True)
        self.cov2 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation2, bias=True)
        self.cov3 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding, dilation3, bias=True)
        
        
        self.norm = nn.LayerNorm(in_chn, eps=1e-6)  
        
        self.act1 = RPReLU(in_chn)
        self.act2 = RPReLU(in_chn) 
        self.act3 = RPReLU(in_chn)

    def forward(self, x):
        B,C,H,W = x.shape
        x = self.move(x)
        x1 = self.cov1(x).permute(0, 2, 3, 1).flatten(1,2)
        x1 = self.act1(x1)
        x2 = self.cov2(x).permute(0, 2, 3, 1).flatten(1,2)
        x2 = self.act2(x2) 
        x3 = self.cov3(x).permute(0, 2, 3, 1).flatten(1,2)
        x3 = self.act3(x3) 
        x = self.norm(x1+x2+x3)
        return x.permute(0, 2, 1).view(-1, C, H, W).contiguous()
    
class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])
    def forward(self, x):
        out = self.block(x)
        return out

class SpatialFeatureEnhancementBlock(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out
    
    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()
    
class InvertedResidual_RAB(nn.Module):
    """Inverted Residual Block.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        mid_channels (int): The input channels of the depthwise convolution.
        kernel_size (int): The kernel size of the depthwise convolution.
            Default: 3.
        groups (None or int): The group number of the depthwise convolution.
            Default: None, which means group number = mid_channels.
        stride (int): The stride of the depthwise convolution. Default: 1.
        se_cfg (dict): Config dict for se layer. Default: None, which means no
            se layer.
        with_expand_conv (bool): Use expand conv or not. If set False,
            mid_channels must be the same with in_channels.
            Default: True.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size=3,
                 groups=None,
                 stride=1,
                 se_cfg=None,
                 with_expand_conv=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        super().__init__()
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.with_se = se_cfg is not None
        self.with_expand_conv = with_expand_conv
        
        if groups is None:
            groups = mid_channels

        if self.with_se:
            assert isinstance(se_cfg, dict)
        if not self.with_expand_conv:
            assert mid_channels == in_channels

        if self.with_expand_conv:
            self.expand_conv = ConvModule(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.depthwise_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if self.with_se:
            self.se = SELayer(**se_cfg)
        self.linear_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.size = 3  # Gaussian kernel size
        self.sigma = 1.0  # Standard deviation for Gaussian
        self.norm_layer = dict(type='BN', requires_grad=True)  # Normalization type
        self.act_layer = nn.ReLU  # Activation function
        self.sfeb = SpatialFeatureEnhancementBlock(dim=mid_channels, size=self.size, sigma=self.sigma, norm_layer=self.norm_layer, act_layer=self.act_layer, feature_extra=True)
        self.msgdc = RAB(in_chn=out_channels, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1, padding='same')
    def forward(self, x):

        def _inner_forward(x):
            out = x

            if self.with_expand_conv:
                out = self.expand_conv(out)

            out = self.depthwise_conv(out)
            out = self.sfeb(out)

            if self.with_se:
                out = self.se(out)

            out = self.linear_conv(out)
            out = self.RAB(out)

            if self.with_res_shortcut:
                return x + out
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out
