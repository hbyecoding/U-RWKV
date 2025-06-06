import torch
import torch.nn as nn
from timm.models.layers import DropPath

# Copyright (c) Shanghai AI Lab. All rights reserved.
from typing import Sequence
import math, os

import torch.utils.checkpoint as cp

from torch_dwconv import DepthwiseConv2d


from torch_dwconv import DepthwiseConv2d

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
import torch
import torch.nn as nn

# 定义默认的激活函数
default_act = nn.GELU()

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act='gelu'):
        super().__init__()
        p = (k // 2) if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        
        # 根据 act 的类型创建激活函数
        if isinstance(act, str):
            if act.lower() == 'gelu':
                self.act = nn.GELU()
            elif act.lower() == 'relu':
                self.act = nn.ReLU()
            elif act.lower() == 'leakyrelu':
                self.act = nn.LeakyReLU(0.1)
            elif act.lower() == 'silu':
                self.act = nn.SiLU()
            else:
                raise ValueError(f"Unsupported activation function: {act}")
        # else:
        #     self.act = default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class DSWConv2d_decoder(nn.Module):
    def __init__(self, ch_in, ch_out, groups=2, ks=3, padding=0, act='gelu',bias = False):
        super().__init__()
        self.dwconv2d = Residual(nn.Sequential(
            DepthwiseConv2d(ch_in, ch_in, kernel_size=ks, padding=padding, groups=groups, bias=bias),
            nn.GELU() if act.lower() == 'gelu' else nn.ReLU() if act.lower() == 'relu' else nn.LeakyReLU(0.1) if act.lower() == 'leakyrelu' else nn.SiLU() if act.lower() == 'silu' else nn.Identity(),
            nn.BatchNorm2d(ch_in)
        ))
        
        # SeperateNconv
        self.conv1x1_expand = nn.Sequential(
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),  
            nn.GELU()if act.lower() == 'gelu' else nn.ReLU() if act.lower() == 'relu' else nn.LeakyReLU(0.1) if act.lower() == 'leakyrelu' else nn.SiLU() if act.lower() == 'silu' else nn.Identity(),
            nn.BatchNorm2d(ch_out* 4),
            
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)), 
            nn.GELU()if act.lower() == 'gelu' else nn.ReLU() if act.lower() == 'relu' else nn.LeakyReLU(0.1) if act.lower() == 'leakyrelu' else nn.SiLU() if act.lower() == 'silu' else nn.Identity(),
            nn.BatchNorm2d(ch_out)
        )


    def forward(self, x):
        # DW
        x = self.dwconv2d(x)
        # 
        x = self.conv1x1_expand(x)
        
        return x
    
    
class DSWConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, groups, ks=3, padding=0, act='gelu',bias = False):
        super().__init__()
        self.dwconv2d = Residual(nn.Sequential(
            DepthwiseConv2d(ch_in, ch_in, kernel_size=ks, padding=padding, groups=groups, bias=bias),
            nn.GELU() if act.lower() == 'gelu' else nn.ReLU() if act.lower() == 'relu' else nn.LeakyReLU(0.1) if act.lower() == 'leakyrelu' else nn.SiLU() if act.lower() == 'silu' else nn.Identity(),
            nn.BatchNorm2d(ch_in)
        ))
        
        # Seperate
        self.conv1x1_expand = nn.Sequential(
            nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1), bias=True),  # 通道扩展
            nn.GELU()if act.lower() == 'gelu' else nn.ReLU() if act.lower() == 'relu' else nn.LeakyReLU(0.1) if act.lower() == 'leakyrelu' else nn.SiLU() if act.lower() == 'silu' else nn.Identity(),
            nn.BatchNorm2d(ch_in * 4)
        )
        self.conv1x1_compress = nn.Sequential(
            nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1), bias=True),  # 通道压缩
            nn.GELU()if act.lower() == 'gelu' else nn.ReLU() if act.lower() == 'relu' else nn.LeakyReLU(0.1) if act.lower() == 'leakyrelu' else nn.SiLU() if act.lower() == 'silu' else nn.Identity(),
            nn.BatchNorm2d(ch_in)
        )


    def forward(self, x):
        # DW
        x = self.dwconv2d(x)
        # 
        x = self.conv1x1_expand(x)

        x = self.conv1x1_compress(x)
        
        return x

class LoANBlock(nn.Module):
    def __init__(self, ch_in, ch_out, N=8, depth=8, shortcut=True,k=3, sep_num=2, act='gelu'):
        """
        e 代表，channels 被隔着取的份数。e = 1/2 代表channels 被分成两个 奇数组和偶数组。e = 1/3 则是 0，3，6，... 为第0组，1，4，7... 为第1组，以此类推 
        """
        super().__init__()
        self.N = N
        # self.is_in_decoder = is_in_decoder,is_in_decoder = False
        self.c = int(ch_out / sep_num /self.N)
        self.add = shortcut and ch_in == ch_out

        self.pwconv1 = Conv(ch_in, ch_out // self.N, 1, 1, act=act)
        self.pwconv2 = Conv(ch_out // sep_num, ch_out, 1, 1, act=act)
        self.m = nn.ModuleList([DSWConv2d(self.c,self.c,groups= self.c, ks=k, padding=k//2, act=act) for _ in range(self.N -1)])
        # if is_in_decoder:
        #     self.de_conv = nn.Sequential(
        #         DSWConv2d(ch_in, ch_out, groups= 2, ks=3, padding=1, bias=True)
        #     )
    def forward(self, x):
        x_residual = x
        x = self.pwconv1(x)

        x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        x.extend(m(x[-1]) for m in self.m)
        x[0] = x[0] + x[1]
        x.pop(1)

        y = torch.cat(x, dim=1)
        y = self.pwconv2(y)
        return x_residual + y if self.add else y
    
    
# block = LoANBlock(128, 128).cuda()
# input = torch.zeros((2, 128, 64, 64)).cuda()
# output = block(input)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), #nearest 上采样 
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)   
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
    
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), #nearest 上采样 
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
        
    
class LoRD(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128,256,768], depths=[8, 8, 8, 8, 8], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(LoRD, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = LoANBlock(ch_in=dims[0], ch_out=dims[0], N=depths[0], k=kernels[0])
        self.encoder2 = LoANBlock(ch_in=dims[0], ch_out=dims[1], N=depths[1], k=kernels[1])
        self.encoder3 = LoANBlock(ch_in=dims[1], ch_out=dims[2], N=depths[2], k=kernels[2])
        self.encoder4 = LoANBlock(ch_in=dims[2], ch_out=dims[3], N=depths[3], k=kernels[3])
        self.encoder5 = LoANBlock(ch_in=dims[3], ch_out=dims[4], N=depths[4], k=kernels[4])
        # # Decoder
        # self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        # self.Up_conv5 = DSWConv2d_decoder(ch_in=dims[3] * 2, ch_out=dims[3],groups=dims[3]*2)
        # self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        # self.Up_conv4 = DSWConv2d_decoder(ch_in=dims[2] * 2, ch_out=dims[2],groups=dims[2]*2)
        # self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        # self.Up_conv3 = DSWConv2d_decoder(ch_in=dims[1] * 2, ch_out=dims[1],groups=dims[1]*2)
        # self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        # self.Up_conv2 = DSWConv2d_decoder(ch_in=dims[0] * 2, ch_out=dims[0],groups=dims[0]*2)
        # self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1

class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x

import torch.nn.functional as F
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcls.models.utils import resize_pos_embed
from mmcls.models.necks.gap import GlobalAveragePooling
from mmcls.models.backbones.base_backbone import BaseBackbone
T_MAX = 1024 #128*128 2048 均不可以 # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


wkv_cuda = load(name="wkv", sources=[os.path.join(os.getcwd(),'Tan9','src','network', 'conv_based','transformer_learning', 'cuda', 'wkv_op.cpp'), os.path.join(os.getcwd(),'Tan9','src','network', 'conv_based', 'transformer_learning', 'cuda', 'wkv_cuda.cu')],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])
    
def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    if patch_resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            patch_resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError("Cannot infer a valid patch_resolution for the given input shape.")

    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    input = input
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)

class VRWKV_SpatialMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, init_mode='fancy', k_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if k_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.value.scale_init = 1

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                
                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, patch_resolution)
        rwkv = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv


class VRWKV_ChannelMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 k_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if k_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

        self.key.scale_init = 1

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv

class VRWKV_Bottleneck(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 drop_path=0., k_norm=True):
        """
        Bottleneck module for (B, N, C) input shape.
        
        Args:
            n_embd (int): Embedding dimension (C).
            n_layer (int): Total number of layers in the model.
            layer_id (int): Current layer ID.
            shift_mode (str): Shift mode for spatial mixing.
            channel_gamma (float): Gamma for channel mixing.
            shift_pixel (int): Number of pixels to shift.
            hidden_rate (int): Hidden dimension multiplier for FFN.
            init_mode (str): Initialization mode ('fancy', 'local', 'global').
            drop_path (float): Drop path rate.
            k_norm (bool): Whether to use key normalization.
        """
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.flatten = nn.Flatten(2)
        # Spatial Mixing (Attention-like mechanism)
        self.spatial_mix = VRWKV_SpatialMix(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            shift_mode=shift_mode,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel,
            init_mode=init_mode,
            k_norm=k_norm
        )

        # Channel Mixing (FFN-like mechanism)
        self.channel_mix = VRWKV_ChannelMix(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            shift_mode=shift_mode,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel,
            hidden_rate=hidden_rate,
            init_mode=init_mode,
            k_norm=k_norm
        )

        # LayerNorm for post-processing
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, patch_resolution=None):
        """
        Forward pass for the bottleneck.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            patch_resolution (tuple): Resolution of the patches (H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, N, C).
        """
        # Spatial Mixing (Attention-like)
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1,2)
        x = x + self.drop_path(self.spatial_mix(self.ln1(x), patch_resolution))

        # Channel Mixing (FFN-like)
        x = x + self.drop_path(self.channel_mix(self.ln2(x), patch_resolution))
        
        if len(x.shape) == 3:
            out = x.transpose(1, 2).reshape(B, C, H, W)
        return out


class VRWKVEncoderBlock(nn.Module):
    def __init__(self, n_embd, n_layer=12, layer_id=0, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 drop_path=0., k_norm=True, depth=1, ch_in=None, ch_out=None):
        """
        Encoder block using VRWKV_Bottleneck layers.
        
        Args:
            n_embd (int): Embedding dimension (C).
            n_layer (int): Total number of layers in the model.
            layer_id (int): Current layer ID.
            shift_mode (str): Shift mode for spatial mixing.
            channel_gamma (float): Gamma for channel mixing.
            shift_pixel (int): Number of pixels to shift.
            hidden_rate (int): Hidden dimension multiplier for FFN.
            init_mode (str): Initialization mode ('fancy', 'local', 'global').
            drop_path (float): Drop path rate.
            k_norm (bool): Whether to use key normalization.
            depth (int): Number of VRWKV_Bottleneck layers.
            ch_in (int): Input channel dimension.
            ch_out (int): Output channel dimension.
        """
        super().__init__()
        self.depth = depth
        self.ch_in = ch_in if ch_in is not None else n_embd
        self.ch_out = ch_out if ch_out is not None else n_embd

        # Sequence of VRWKV_Bottleneck layers
        self.bottleneck_layers = nn.ModuleList([
            VRWKV_Bottleneck(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id + i,
                shift_mode=shift_mode,
                channel_gamma=channel_gamma,
                shift_pixel=shift_pixel,
                hidden_rate=hidden_rate,
                init_mode=init_mode,
                drop_path=drop_path,
                k_norm=k_norm
            ) for i in range(depth)
        ])

        # Convolutional block for upsampling or downsampling
        self.conv_block = conv_block(self.ch_in, self.ch_out)

    def forward(self, x, patch_resolution=None):
        """
        Forward pass for the encoder block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            patch_resolution (tuple): Resolution of the patches (H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, N, C_out).
        """
        for layer in self.bottleneck_layers:
            x = layer(x, patch_resolution)

        # Reshape if necessary (e.g., from (B, N, C) to (B, C, H, W))
        if len(x.shape) == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)

        # Apply the convolutional block
        x = self.conv_block(x)

        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
    
    
class LoRD_4plusDeep(nn.Module):
    def __init__(self, input_channel=3, num_classes=1,
                 drop_path_rate=0.1,layer_scale_init_value=1e-6, head_init_scale=1.,
                 dims=[16, 32, 128,256,768], depths=[3, 3, 3, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        # self.encoder1 = LoANBlock(ch_in=dims[0], ch_out=dims[0], N=depths[0], k=kernels[0])
        # self.encoder2 = LoANBlock(ch_in=dims[0], ch_out=dims[1], N=depths[1], k=kernels[1])
        # self.encoder3 = LoANBlock(ch_in=dims[1], ch_out=dims[2], N=depths[2], k=kernels[2])
        # self.encoder4 = LoANBlock(ch_in=dims[2], ch_out=dims[3], N=depths[3], k=kernels[3])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        # self.encoder4 = VRWKVEncoderBlock(
        #     n_embd=192,
        #     depth=depths[3]
        # )
        # self.stages = nn.ModuleList() # 5 feature resolution stages, each consisting of multiple residual blocks
        # dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        # cur = 0
        # for i in range(len(depths)):
        #     stage = nn.Sequential(
        #         *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
        #         layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
        #     )
        #     self.stages.append(stage)
        #     cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        # # self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)
        self.encoder5 = nn.Sequential(
            Block(dim=dims[3], drop_path= drop_path_rate),
            conv_block(ch_in=dims[3], ch_out=dims[4])
        )
        # deep5_layer = Block(dims[-2],drop_path=0.01)
        # self.deep5_layers = nn.ModuleList()
        # for i in range(depths[-1]):
        #     self.deep5_layers.append(deep5_layer)
        
        
        # self.encoder5 = LoANBlock(ch_in=dims[3], ch_out=dims[4], N=depths[4], k=kernels[4])
        # # Decoder
        # self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        # self.Up_conv5 = DSWConv2d_decoder(ch_in=dims[3] * 2, ch_out=dims[3],groups=dims[3]*2)
        # self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        # self.Up_conv4 = DSWConv2d_decoder(ch_in=dims[2] * 2, ch_out=dims[2],groups=dims[2]*2)
        # self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        # self.Up_conv3 = DSWConv2d_decoder(ch_in=dims[1] * 2, ch_out=dims[1],groups=dims[1]*2)
        # self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        # self.Up_conv2 = DSWConv2d_decoder(ch_in=dims[0] * 2, ch_out=dims[0],groups=dims[0]*2)
        # self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1

class LoRD_128_192_384(nn.Module):
    def __init__(self, input_channel=3, num_classes=1,
                 drop_path_rate=0.1,layer_scale_init_value=1e-6, head_init_scale=1.,
                 dims=[16, 32,128, 192,384], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        # self.encoder1 = LoANBlock(ch_in=dims[0], ch_out=dims[0], N=depths[0], k=kernels[0])
        # self.encoder2 = LoANBlock(ch_in=dims[0], ch_out=dims[1], N=depths[1], k=kernels[1])
        # self.encoder3 = LoANBlock(ch_in=dims[1], ch_out=dims[2], N=depths[2], k=kernels[2])
        # self.encoder4 = LoANBlock(ch_in=dims[2], ch_out=dims[3], N=depths[3], k=kernels[3])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        # self.encoder4 = nn.Sequential(
        #     VRWKVEncoderBlock(
        #         n_embd=dims[2],
        #         depth=depths[2]
        #     ),
        #     conv_block(ch_in=dims[2], ch_out=dims[3])            
        # )
        # self.encoder5 = nn.Sequential(
        #     VRWKVEncoderBlock(
        #         n_embd=dims[3],
        #         depth=depths[3]
        #     ),
        #     conv_block(ch_in=dims[3], ch_out=dims[4])            
        # )
        self.encoder5 = CMUNeXtBlock(dims[3], dims[4], depths[4], kernels[4])

        # self.stages = nn.ModuleList() # 5 feature resolution stages, each consisting of multiple residual blocks
        # dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        # cur = 0
        # for i in range(len(depths)):
        #     stage = nn.Sequential(
        #         *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
        #         layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
        #     )
        #     self.stages.append(stage)
        #     cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        # # self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)
        # self.encoder5 = nn.Sequential(
        #     Block(dim=dims[3], drop_path= drop_path_rate),
        #     conv_block(ch_in=dims[3], ch_out=dims[4])
        # )
        # deep5_layer = Block(dims[-2],drop_path=0.01)
        # self.deep5_layers = nn.ModuleList()
        # for i in range(depths[-1]):
        #     self.deep5_layers.append(deep5_layer)
        self.rwkv_dim2 = VRWKV_Bottleneck(n_embd=dims[2],n_layer=12,layer_id=0,
                                     shift_mode='q_shift',
                                     channel_gamma=1/4,
                                     shift_pixel=1,
                                     hidden_rate=4,
                                     drop_path=0.1,
                                     init_mode='fancy',
                                     k_norm=True)      
        self.rwkv_dim3 = VRWKV_Bottleneck(n_embd=dims[3],n_layer=12,layer_id=0,
                                     shift_mode='q_shift',
                                     channel_gamma=1/4,
                                     shift_pixel=1,
                                     hidden_rate=4,
                                     drop_path=0.1,
                                     init_mode='fancy',
                                     k_norm=True)      
        self.conv_blk_dim2_3 = conv_block(ch_in=dims[2],ch_out=dims[3])  
        self.conv_blk_dim3_4 = conv_block(ch_in=dims[3],ch_out=dims[4])  
        self.rwkv_bottleneck = VRWKV_Bottleneck(n_embd=dims[4],n_layer=12,layer_id=0,
                                     shift_mode='q_shift',
                                     channel_gamma=1/4,
                                     shift_pixel=1,
                                     hidden_rate=4,
                                     drop_path=0.1,
                                     init_mode='fancy',
                                     k_norm=True)
        
        # self.encoder5 = LoANBlock(ch_in=dims[3], ch_out=dims[4], N=depths[4], k=kernels[4])
        # # Decoder
        # self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        # self.Up_conv5 = DSWConv2d_decoder(ch_in=dims[3] * 2, ch_out=dims[3],groups=dims[3]*2)
        # self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        # self.Up_conv4 = DSWConv2d_decoder(ch_in=dims[2] * 2, ch_out=dims[2],groups=dims[2]*2)
        # self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        # self.Up_conv3 = DSWConv2d_decoder(ch_in=dims[1] * 2, ch_out=dims[1],groups=dims[1]*2)
        # self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        # self.Up_conv2 = DSWConv2d_decoder(ch_in=dims[0] * 2, ch_out=dims[0],groups=dims[0]*2)
        # self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        # x4 = self.encoder4(x4)
        # x5 = self.Maxpool(x4)
        x4_ = self.rwkv_dim2(x4)
        x4 = self.conv_blk_dim2_3(x4_)
        
        # x5 = self.encoder5(x5)
        x5_ = self.rwkv_dim3(x5)
        x5 = self.conv_blk_dim3_4(x5_)
        
        
        x5 = self.rwkv_bottleneck(x5)
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1
    
class LoRD_128_192_384_enc5_bot(nn.Module):
    def __init__(self, input_channel=3, num_classes=1,
                 drop_path_rate=0.1,layer_scale_init_value=1e-6, head_init_scale=1.,
                 dims=[16, 32,128, 192,384], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        # self.encoder1 = LoANBlock(ch_in=dims[0], ch_out=dims[0], N=depths[0], k=kernels[0])
        # self.encoder2 = LoANBlock(ch_in=dims[0], ch_out=dims[1], N=depths[1], k=kernels[1])
        # self.encoder3 = LoANBlock(ch_in=dims[1], ch_out=dims[2], N=depths[2], k=kernels[2])
        # self.encoder4 = LoANBlock(ch_in=dims[2], ch_out=dims[3], N=depths[3], k=kernels[3])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        # self.encoder4 = nn.Sequential(
        #     VRWKVEncoderBlock(
        #         n_embd=dims[2],
        #         depth=depths[2]
        #     ),
        #     conv_block(ch_in=dims[2], ch_out=dims[3])            
        # )
        # self.encoder5 = nn.Sequential(
        #     VRWKVEncoderBlock(
        #         n_embd=dims[3],
        #         depth=depths[3]
        #     ),
        #     conv_block(ch_in=dims[3], ch_out=dims[4])            
        # )
        self.encoder5 = CMUNeXtBlock(dims[3], dims[4], depths[4], kernels[4])

        # self.stages = nn.ModuleList() # 5 feature resolution stages, each consisting of multiple residual blocks
        # dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        # cur = 0
        # for i in range(len(depths)):
        #     stage = nn.Sequential(
        #         *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
        #         layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
        #     )
        #     self.stages.append(stage)
        #     cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        # # self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)
        # self.encoder5 = nn.Sequential(
        #     Block(dim=dims[3], drop_path= drop_path_rate),
        #     conv_block(ch_in=dims[3], ch_out=dims[4])
        # )
        # deep5_layer = Block(dims[-2],drop_path=0.01)
        # self.deep5_layers = nn.ModuleList()
        # for i in range(depths[-1]):
        #     self.deep5_layers.append(deep5_layer)
        self.rwkv_dim2 = VRWKV_Bottleneck(n_embd=dims[2],n_layer=12,layer_id=0,
                                     shift_mode='q_shift',
                                     channel_gamma=1/4,
                                     shift_pixel=1,
                                     hidden_rate=4,
                                     drop_path=0.1,
                                     init_mode='fancy',
                                     k_norm=True)      
        self.rwkv_dim3 = VRWKV_Bottleneck(n_embd=dims[3],n_layer=12,layer_id=0,
                                     shift_mode='q_shift',
                                     channel_gamma=1/4,
                                     shift_pixel=1,
                                     hidden_rate=4,
                                     drop_path=0.1,
                                     init_mode='fancy',
                                     k_norm=True)      
        self.conv_blk_dim2_3 = conv_block(ch_in=dims[2],ch_out=dims[3])  
        self.conv_blk_dim3_4 = conv_block(ch_in=dims[3],ch_out=dims[4])  
        self.rwkv_bottleneck = VRWKV_Bottleneck(n_embd=dims[4],n_layer=12,layer_id=0,
                                     shift_mode='q_shift',
                                     channel_gamma=1/4,
                                     shift_pixel=1,
                                     hidden_rate=4,
                                     drop_path=0.1,
                                     init_mode='fancy',
                                     k_norm=True)
        
        # self.encoder5 = LoANBlock(ch_in=dims[3], ch_out=dims[4], N=depths[4], k=kernels[4])
        # # Decoder
        # self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        # self.Up_conv5 = DSWConv2d_decoder(ch_in=dims[3] * 2, ch_out=dims[3],groups=dims[3]*2)
        # self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        # self.Up_conv4 = DSWConv2d_decoder(ch_in=dims[2] * 2, ch_out=dims[2],groups=dims[2]*2)
        # self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        # self.Up_conv3 = DSWConv2d_decoder(ch_in=dims[1] * 2, ch_out=dims[1],groups=dims[1]*2)
        # self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        # self.Up_conv2 = DSWConv2d_decoder(ch_in=dims[0] * 2, ch_out=dims[0],groups=dims[0]*2)
        # self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)

        x5 = self.Maxpool(x4)
        # x5 = self.encoder5(x5)
        x5_ = self.rwkv_dim3(x5)
        x5 = self.conv_blk_dim3_4(x5_)
        
        
        x5 = self.rwkv_bottleneck(x5)
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1

class LoRD_double_192_384_768(nn.Module):
    def __init__(self, input_channel=3, num_classes=1,
                 drop_path_rate=0.1,layer_scale_init_value=1e-6, head_init_scale=1.,
                 dims=[16, 32, 192,384,768], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        # self.encoder1 = LoANBlock(ch_in=dims[0], ch_out=dims[0], N=depths[0], k=kernels[0])
        # self.encoder2 = LoANBlock(ch_in=dims[0], ch_out=dims[1], N=depths[1], k=kernels[1])
        # self.encoder3 = LoANBlock(ch_in=dims[1], ch_out=dims[2], N=depths[2], k=kernels[2])
        # self.encoder4 = LoANBlock(ch_in=dims[2], ch_out=dims[3], N=depths[3], k=kernels[3])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        # self.encoder4 = nn.Sequential(
        #     VRWKVEncoderBlock(
        #         n_embd=dims[2],
        #         depth=depths[2]
        #     ),
        #     conv_block(ch_in=dims[2], ch_out=dims[3])            
        # )
        # self.encoder5 = nn.Sequential(
        #     VRWKVEncoderBlock(
        #         n_embd=dims[3],
        #         depth=depths[3]
        #     ),
        #     conv_block(ch_in=dims[3], ch_out=dims[4])            
        # )
        self.encoder5 = CMUNeXtBlock(dims[3], dims[4], depths[4], kernels[4])

        # self.stages = nn.ModuleList() # 5 feature resolution stages, each consisting of multiple residual blocks
        # dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        # cur = 0
        # for i in range(len(depths)):
        #     stage = nn.Sequential(
        #         *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
        #         layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
        #     )
        #     self.stages.append(stage)
        #     cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        # # self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)
        # self.encoder5 = nn.Sequential(
        #     Block(dim=dims[3], drop_path= drop_path_rate),
        #     conv_block(ch_in=dims[3], ch_out=dims[4])
        # )
        # deep5_layer = Block(dims[-2],drop_path=0.01)
        # self.deep5_layers = nn.ModuleList()
        # for i in range(depths[-1]):
        #     self.deep5_layers.append(deep5_layer)
        self.rwkv = VRWKV_Bottleneck(n_embd=dims[4],n_layer=12,layer_id=0,
                                     shift_mode='q_shift',
                                     channel_gamma=1/4,
                                     shift_pixel=1,
                                     hidden_rate=4,
                                     drop_path=0.1,
                                     init_mode='fancy',
                                     k_norm=True)
        
        # self.encoder5 = LoANBlock(ch_in=dims[3], ch_out=dims[4], N=depths[4], k=kernels[4])
        # # Decoder
        # self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        # self.Up_conv5 = DSWConv2d_decoder(ch_in=dims[3] * 2, ch_out=dims[3],groups=dims[3]*2)
        # self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        # self.Up_conv4 = DSWConv2d_decoder(ch_in=dims[2] * 2, ch_out=dims[2],groups=dims[2]*2)
        # self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        # self.Up_conv3 = DSWConv2d_decoder(ch_in=dims[1] * 2, ch_out=dims[1],groups=dims[1]*2)
        # self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        # self.Up_conv2 = DSWConv2d_decoder(ch_in=dims[0] * 2, ch_out=dims[0],groups=dims[0]*2)
        # self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        x5 = self.rwkv(x5)
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1
    
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

if __name__ == "__main__":
    x = torch.randn(3,3, 256, 256)
    x = x.cuda()
    # model = LoRD_4plusDeep(dims=[16, 32, 128, 192, 384])
    # model = LoRD_double_192_384_768().cuda()
    model = LoRD_128_192_384().cuda()
    print(count_params(model))
    
    out = model(x)
    print(out.shape)
        