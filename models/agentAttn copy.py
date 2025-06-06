# -*- coding: utf-8 -*-
# fasternet.py
# author: lm

"""
https://arxiv.org/abs/2303.03667
<<Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks>>
"""

from collections import OrderedDict
from functools import partial
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from timm.models.layers import DropPath

from pconv import PConv2d


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size:int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 act: str = 'ReLU'):
        super(ConvBNLayer, self).__init__()
        assert act in ('ReLU', 'GELU')
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = getattr(nn, act)()
     
    def _fuse_bn_tensor(self) -> None:
        kernel = self.conv.weight 
        bias = self.conv.bias if hasattr(self.conv, 'bias') and self.conv.bias is not None else 0
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var 
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps 
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        self.conv.weight.data = kernel * t 
        self.conv.bias = nn.Parameter(beta - (running_mean - bias) * gamma / std, requires_grad=False)
        self.bn = nn.Identity()
        return self.conv.weight.data, self.conv.bias.data 
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x 
        
        
class FasterNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 inner_channels: int = None,
                 kernel_size: int = 3,
                 bias=False,
                 act: str = 'ReLU',
                 n_div:int = 4,
                 forward: str = 'split_cat',
                 drop_path: float = 0.,
                 ):
        super(FasterNetBlock, self).__init__()
        inner_channels = inner_channels or in_channels * 2
        self.conv1 = PConv2d(in_channels,
                             kernel_size,
                             n_div,
                             forward)
        self.conv2 = ConvBNLayer(in_channels,
                                 inner_channels,
                                 bias=bias,
                                 act=act)
        self.conv3 = nn.Conv2d(inner_channels,
                               in_channels,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
 
    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        
        return x + self.drop_path(y) 



class FasterNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels = 1000,
                 last_channels = 1280,
                 inner_channels: list = [40, 80, 160, 320],
                 blocks: list = [1, 2, 8, 2],
                 bias=False,
                 act='ReLU',
                 n_div = 4,
                 forward = 'slicing',
                 drop_path = 0.,
                 ):
        super(FasterNet, self).__init__()
        self.embedding = ConvBNLayer(in_channels,
                                     inner_channels[0],
                                     kernel_size=4,
                                     stride=4,
                                     bias=bias)
        
        self.stage1 = nn.Sequential(OrderedDict([
                ('block{}'.format(idx), 
                 FasterNetBlock(inner_channels[0],
                                bias=bias,
                                act=act,
                                n_div=n_div,
                                forward=forward,
                                drop_path=drop_path)) for idx in range(blocks[0])]))

        self.merging1 = ConvBNLayer(inner_channels[0],
                                inner_channels[1],
                                kernel_size=2,
                                stride=2,
                                bias=bias)
        
        self.stage2 = nn.Sequential(OrderedDict([
                ('block{}'.format(idx), 
                 FasterNetBlock(inner_channels[1],
                                bias=bias,
                                act=act,
                                n_div=n_div,
                                forward=forward,
                                drop_path=drop_path)) for idx in range(blocks[1])]))

        self.merging2 = ConvBNLayer(inner_channels[1],
                                inner_channels[2],
                                kernel_size=2,
                                stride=2,
                                bias=bias)
        
        self.stage3 = nn.Sequential(OrderedDict([
                ('block{}'.format(idx), 
                 FasterNetBlock(inner_channels[2],
                                bias=bias,
                                act=act,
                                n_div=n_div,
                                forward=forward,
                                drop_path=drop_path)) for idx in range(blocks[2])]))

        self.merging3 = ConvBNLayer(inner_channels[2],
                                inner_channels[3],
                                kernel_size=2,
                                stride=2,
                                bias=bias)
        
        self.stage4 = nn.Sequential(OrderedDict([
                ('block{}'.format(idx), 
                 FasterNetBlock(inner_channels[3],
                                bias=bias,
                                act=act,
                                n_div=n_div,
                                forward=forward,
                                drop_path=drop_path)) for idx in range(blocks[3])]))

        self.classifier = nn.Sequential(OrderedDict([
            ('global_average_pooling', nn.AdaptiveAvgPool2d(1)),
            ('conv', nn.Conv2d(inner_channels[-1], last_channels, kernel_size=1, bias=False)),
            ('act', getattr(nn, act)()),
            ('flat', nn.Flatten()),
            ('fc', nn.Linear(last_channels, out_channels, bias=True))
        ]))
        self.feature_channels = inner_channels
    
    
    def fuse_bn_tensor(self):
        for m in self.modules():
            if isinstance(m, ConvBNLayer):
                m._fuse_bn_tensor()
        
        
        
    def forward_feature(self, x: Tensor) -> List[Tensor]:
        x1 = self.stage1(self.embedding(x))
        x2 = self.stage2(self.merging1(x1))
        x3 = self.stage3(self.merging2(x2))
        x4 = self.stage4(self.merging3(x3))
        return [x1, x2, x3, x4]
    
    
    def forward(self, x: Tensor) -> Tensor:
        _, _, _, x = self.forward_feature(x)
        x = self.classifier(x)
        
        return x 
        
        
FasterNetT0 = partial(FasterNet, inner_channels=[40, 80, 160, 320], blocks=[1, 2, 8, 2], act='GELU', drop_path=0.)

FasterNetT1 = partial(FasterNet, inner_channels=[64, 128, 256, 512], blocks=[1, 2, 8, 2], act='GELU', drop_path=0.02)

FasterNetT2 = partial(FasterNet, inner_channels=[96, 192, 384, 768], blocks=[1, 2, 8, 2], act='ReLU', drop_path=0.05)

FasterNetS = partial(FasterNet, inner_channels=[128, 256, 512, 1024], blocks=[1, 2, 13, 2], act='ReLU', drop_path=0.1)

FasterNetM = partial(FasterNet, inner_channels=[144, 288, 576, 1152], blocks=[3, 4, 18, 3], act='ReLU', drop_path=0.2)

FasterNetL = partial(FasterNet, inner_channels=[192, 384, 768, 1536], blocks=[3, 4, 18, 3], act='ReLU', drop_path=0.3)



if __name__ == "__main__":
    import time 
    device = torch.device("cpu")
    
    model = FasterNetL(drop_path=0.)
    
    model = model.to(device)
    print(model)
    model.eval()
    
    x = torch.randn((1, 3, 224, 224), device=device)
    y1 = model(x)
    
    model.fuse_bn_tensor()
    y2 = model(x)
    
    print(y1)
    print(y2)
    print(torch.norm(y1 - y2))
    
    st = time.time()
    test_round = 100
    for i in range(test_round):
        y = model(x)
    et = time.time()
    print("avg time: {}".format((et - st) / test_round))
    print(y.size())
    # Ryzen 2600 CPU 203ms.
    
    