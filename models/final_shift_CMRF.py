# -*- encoding: utf-8 -*-
'''
@File   :  Untitled-11
@Time   :  2024/12/12 23:25:07
@Author :  hbye
'''
######################################   外部调用   ######################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
#########################################################################################
######################################   内部调用   ######################################
#########################################################################################



import math
from thop import clever_format, profile
from torchsummary import summary

class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False) 
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 
        

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias = False) 
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x) 
        # import pdb 
        # pdb.set_trace() 
        
        
        out = self.alpha[0]*x + self.alpha[1]*out1x1 + self.alpha[2]*out3x3 + self.alpha[3]*out5x5
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution 
        
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2)) 
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1)) 
        
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2)) 
        
        combined_weight = self.alpha[0]*identity_weight + self.alpha[1]*padded_weight_1x1 + self.alpha[2]*padded_weight_3x3 + self.alpha[3]*self.conv5x5.weight 
        
        device = self.conv5x5_reparam.weight.device 

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)


    def forward(self, x): 
        
        if self.training: 
            self.repram_flag = True
            out = self.forward_train(x) 
        elif self.training == False and self.repram_flag == True:
            self.reparam_5x5() 
            self.repram_flag = False 
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv5x5_reparam(x)
        
        return out 
    
    

def autopad(k, p=None, d=1):  
    '''
    k: kernel
    p: padding
    d: dilation
    '''
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k] # actual kernel-size
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, paddin g, groups, dilation, activation)."""
    default_act = nn.GELU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution with args(ch_in, ch_out, kernel, stride, dilation, activation)."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

# 实例化


    
# Lightweight Cascade Multi-Receptive Fields Module
class CMRF(nn.Module):
    """CMRF Module with args(ch_in, ch_out, number, shortcut, groups, expansion)."""
    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        super().__init__()
        
        self.N         = N
        self.c         = int(c2 * e / self.N)
        self.add       = shortcut and c1 == c2
        
        self.pwconv1   = Conv(c1, c2//self.N, 1, 1)
        self.pwconv2   = Conv(c2//2, c2, 1, 1)
        self.m         = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N-1))

    def forward(self, x):
        """Forward pass through CMRF Module."""
        x_residual = x
        x          = self.pwconv1(x)

        x          = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        x.extend(m(x[-1]) for m in self.m)
        x[0]       = x[0] +  x[1] 
        x.pop(1)
        
        y          = torch.cat(x, dim=1) 
        y          = self.pwconv2(y)
        return x_residual + y if self.add else y
    
    # 
    
    
    
# 自动填充函数
def autopad(k, p=None, d=1):
    """
    k: kernel
    p: padding
    d: dilation
    """
    if d > 1:
        # 实际的卷积核大小
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # 自动填充
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

# 标准卷积模块
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.GELU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

# 深度卷积模块
class DWConv(Conv):
    """Depth-wise convolution with args(ch_in, ch_out, kernel, stride, dilation, activation)."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

# 多尺度多感受野模块
class CMRF(nn.Module):
    """CMRF Module with args(ch_in, ch_out, number, shortcut, groups, expansion)."""
    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        super().__init__()
        
        self.N         = N
        self.c         = int(c2 * e / self.N)
        self.add       = shortcut and c1 == c2
        
        self.pwconv1   = Conv(c1, c2//self.N, 1, 1)
        self.pwconv2   = Conv(c2//2, c2, 1, 1)
        self.m         = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N-1))

    def forward(self, x):
        """Forward pass through CMRF Module."""
        x_residual = x
        x          = self.pwconv1(x)

        x          = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        x.extend(m(x[-1]) for m in self.m)
        x[0]       = x[0] +  x[1] 
        x.pop(1)
        
        y          = torch.cat(x, dim=1) 
        y          = self.pwconv2(y)
        return x_residual + y if self.add else y

# 多尺度多感受野模块
class MSMRShift(nn.Module):
    def __init__(self, dim):
        super(MSMRShift, self).__init__()
        # 定义训练阶段的层
        self.conv1x1 = CMRF(dim, dim, N=1, shortcut=False, e=1.0)
        self.conv3x3 = CMRF(dim, dim, N=1, shortcut=False, e=1.0)
        self.conv5x5 = CMRF(dim, dim, N=1, shortcut=False, e=1.0)
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # 定义测试阶段的层
        self.conv5x5_reparam = CMRF(dim, dim, N=1, shortcut=False, e=1.0)
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        out = self.alpha[0] * x + self.alpha[1] * out1x1 + self.alpha[2] * out3x3 + self.alpha[3] * out5x5
        return out

    def reparam_5x5(self):
        # 将 conv1x1, conv3x3, 和 conv5x5 的参数合并为一个 5x5 深度卷积
        padded_weight_1x1 = F.pad(self.conv1x1.pwconv1.conv.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.pwconv1.conv.weight, (1, 1, 1, 1))
        identity_weight = F.pad(torch.ones_like(self.conv1x1.pwconv1.conv.weight), (2, 2, 2, 2))
        combined_weight = self.alpha[0] * identity_weight + self.alpha[1] * padded_weight_1x1 + self.alpha[2] * padded_weight_3x3 + self.alpha[3] * self.conv5x5.pwconv1.conv.weight
        device = self.conv5x5_reparam.pwconv1.conv.weight.device
        combined_weight = combined_weight.to(device)
        self.conv5x5_reparam.pwconv1.conv.weight = nn.Parameter(combined_weight)

    def forward(self, x):
        if self.training:
            self.repram_flag = True
            out = self.forward_train(x)
        elif not self.training and self.repram_flag:
            self.reparam_5x5()
            self.repram_flag = False
            out = self.conv5x5_reparam(x)
        elif not self.training and not self.repram_flag:
            out = self.conv5x5_reparam(x)
        return out
    

def q_shift(input, shift_pixel=1, gamma=0.25, patch_resolution=None, add_residual=True):
    """
    Q-Shift 操作的 4D 版本，输入形状为 (B, C, H, W)。
    
    Args:
        input: 输入张量，形状为 (B, C, H, W)。
        shift_pixel: 每个方向的移动距离。
        gamma: 每个方向的通道数比例。
        add_residual: 是否添加残差连接，默认为 True。
    
    Returns:
        经过 Q-Shift 操作后的张量，如果 add_residual=True，则带有残差连接。
    """
    assert gamma <= 1/4
    B, N, C = input.shape
    if patch_resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            patch_resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError("Cannot infer a valid patch_resolution for the given input shape.")

    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    m_op = MSMRShift(768)
    B, C, H, W = input.shape
    output = torch.zeros_like(input)

    # 计算每个方向的通道数
    gamma_C = int(C * gamma)

    # # 右移
    # output[:, 0:gamma_C, :, shift_pixel:W] = input[:, 0:gamma_C, :, 0:W-shift_pixel]
    # # 左移
    # output[:, gamma_C:2*gamma_C, :, 0:W-shift_pixel] = input[:, gamma_C:2*gamma_C, :, shift_pixel:W]
    # # 下移
    # output[:, 2*gamma_C:3*gamma_C, shift_pixel:H, :] = input[:, 2*gamma_C:3*gamma_C, 0:H-shift_pixel, :]
    # # 上移
    # output[:, 3*gamma_C:4*gamma_C, 0:H-shift_pixel, :] = input[:, 3*gamma_C:4*gamma_C, shift_pixel:H, :]
    # # 保留剩余通道
    # output[:, 4*gamma_C:, ...] = input[:, 4*gamma_C:, ...]
    
    
    output = m_op(input)
    gamma_ = int(1/gamma)

    # 下移
    # 如果 add_residual=True，则添加残差连接
    if add_residual:
        output = input + output

    return output.flatten(2).transpose(1,2)

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.GELU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    


def q_shift(input,N_layers = 8,  shift_pixel=1,gamma=0.25, patch_resolution=None, add_residual=True):
    """
    Q-Shift 操作的 4D 版本，输入形状为 (B, C, H, W)。
    
    Args:
        input: 输入张量，形状为 (B, C, H, W)。
        shift_pixel: 每个方向的移动距离。
        gamma: 每个方向的通道数比例。
        add_residual: 是否添加残差连接，默认为 True。
    
    Returns:
        经过 Q-Shift 操作后的张量，如果 add_residual=True，则带有残差连接。
    """
    assert gamma <= 1/4 
    _gamma = int(1/gamma)
    B, N, C = input.shape
    if patch_resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            patch_resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError("Cannot infer a valid patch_resolution for the given input shape.")

    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    conv_by = Conv(C, C//N_layers, 1, 1)
    conv_N = Conv(C//N_layers, C, 1, 1)
    B, C, H, W = input.shape
    output = torch.zeros_like(input)


    x_by_n_layer = conv_by(input)
    
    # 右移
    output[:, 0:_gamma, :, shift_pixel:W] = x_by_n_layer[:, 0:_gamma, :, 0:W-shift_pixel]
    # 左移
    output[:, 1:_gamma, :, 0:W-shift_pixel] = x_by_n_layer[:, 1:_gamma, :, shift_pixel:W]
    # 下移
    output[:, 2:_gamma, shift_pixel:H, :] = x_by_n_layer[:, 2:_gamma, 0:H-shift_pixel, :]
    # 上移
    output[:, 3:_gamma, 0:H-shift_pixel, :] = x_by_n_layer[:, 3:_gamma, shift_pixel:H, :]
    # # 保留剩余通道
    # output[:, 4*gamma_C:, ...] = input[:, 4*gamma_C:, ...]

    output_N = conv_N(output)
   

    # 下移
    # 如果 add_residual=True，则添加残差连接
    if add_residual:
        output = input + output_N

    return output.flatten(2).transpose(1,2)


def q_shift(input,N_layers = 8,  shift_pixel=1, gamma=0.25, patch_resolution=None, add_residual=True):
    """
    Q-Shift 操作的 4D 版本，输入形状为 (B, C, H, W)。
    
    Args:
        input: 输入张量，形状为 (B, C, H, W)。
        shift_pixel: 每个方向的移动距离。
        gamma: 每个方向的通道数比例。
        add_residual: 是否添加残差连接，默认为 True。
    
    Returns:
        经过 Q-Shift 操作后的张量，如果 add_residual=True，则带有残差连接。
    """
    assert gamma <= 1/4
    B, N, C = input.shape
    if patch_resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            patch_resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError("Cannot infer a valid patch_resolution for the given input shape.")

    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    conv_by = Conv(C, C//N_layers, 1, 1)
    conv_N = Conv(C//N_layers, C, 1, 1)    
    # 计算每个方向的通道数
    gamma_C = int(C * gamma)

    x_by_n_layer = conv_by(input)

    # 右移
    output[:, 0:gamma_C, :, shift_pixel:W] = x_by_n_layer[:, 0:gamma_C, :, 0:W-shift_pixel]
    # 左移
    output[:, gamma_C:2*gamma_C, :, 0:W-shift_pixel] = x_by_n_layer[:, gamma_C:2*gamma_C, :, shift_pixel:W]
    # 下移
    output[:, 2*gamma_C:3*gamma_C, shift_pixel:H, :] = x_by_n_layer[:, 2*gamma_C:3*gamma_C, 0:H-shift_pixel, :]
    # 上移
    output[:, 3*gamma_C:4*gamma_C, 0:H-shift_pixel, :] = x_by_n_layer[:, 3*gamma_C:4*gamma_C, shift_pixel:H, :]
    # 保留剩余通道
    output[:, 4*gamma_C:, ...] = input[:, 4*gamma_C:, ...]
    
    output_N = conv_N(output)


    # 如果 add_residual=True，则添加残差连接
    if add_residual:
        output = input + output_N

    return output.flatten(2).transpose(1,2)



