import torch
import torch.nn as nn
from timm.models.layers import DropPath

# Copyright (c) Shanghai AI Lab. All rights reserved.
from typing import Sequence
import math, os

import logging
import torch
import torch.nn as nn

import torch.utils.checkpoint as cp

from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcls.models.utils import resize_pos_embed
from mmcls.models.necks.gap import GlobalAveragePooling
from mmcls.models.backbones.base_backbone import BaseBackbone

# from .utils import DropPath

import torch.nn as nn


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


logger = logging.getLogger(__name__)


T_MAX = 640 # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load
# wkv_cuda = load(name="wkv", sources=["models/cuda/wkv_op.cpp", "models/cuda/wkv_cuda.cu"],
#                 verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])
# wkv_cuda = load(name="wkv", sources=["./transformer_learning/cuda/wkv_op.cpp", "./transformer_learning/cuda/wkv_cuda.cu"],
#                                      verbose=True, extra_cuda_cflags=['-res-usage', '--maxregcount 60', '-03', '-Xptxas -03',f'-DTmax={T_MAX}'])
# import MultiScaleDeformableAttention
wkv_cuda = load(name="wkv", sources=[os.path.join(os.getcwd(),'Tan9','src','network', 'conv_based','transformer_learning', 'cuda', 'wkv_op.cpp'), os.path.join(os.getcwd(),'Tan9','src','network', 'conv_based', 'transformer_learning', 'cuda', 'wkv_cuda.cu')],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])
    

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



def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


# def q_shift(input, shift_pixel=1, gamma=0.25, patch_resolution=None, add_residual=True):
#     """
#     Q-Shift 操作的 4D 版本，输入形状为 (B, C, H, W)。
    
#     Args:
#         input: 输入张量，形状为 (B, C, H, W)。
#         shift_pixel: 每个方向的移动距离。
#         gamma: 每个方向的通道数比例。
#         add_residual: 是否添加残差连接，默认为 True。
    
#     Returns:
#         经过 Q-Shift 操作后的张量，如果 add_residual=True，则带有残差连接。
#     """
#     assert gamma <= 1/4
#     B, N, C = input.shape
#     if patch_resolution is None:
#         sqrt_N = int(N ** 0.5)
#         if sqrt_N * sqrt_N == N:
#             patch_resolution = (sqrt_N, sqrt_N)
#         else:
#             raise ValueError("Cannot infer a valid patch_resolution for the given input shape.")

#     input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    
#     B, C, H, W = input.shape
#     output = torch.zeros_like(input)

#     # 计算每个方向的通道数
#     gamma_C = int(C * gamma)

#     # 右移
#     output[:, 0:gamma_C, :, shift_pixel:W] = input[:, 0:gamma_C, :, 0:W-shift_pixel]
#     # 左移
#     output[:, gamma_C:2*gamma_C, :, 0:W-shift_pixel] = input[:, gamma_C:2*gamma_C, :, shift_pixel:W]
#     # 下移
#     output[:, 2*gamma_C:3*gamma_C, shift_pixel:H, :] = input[:, 2*gamma_C:3*gamma_C, 0:H-shift_pixel, :]
#     # 上移
#     output[:, 3*gamma_C:4*gamma_C, 0:H-shift_pixel, :] = input[:, 3*gamma_C:4*gamma_C, shift_pixel:H, :]
#     # 保留剩余通道
#     output[:, 4*gamma_C:, ...] = input[:, 4*gamma_C:, ...]

#     # 如果 add_residual=True，则添加残差连接
#     if add_residual:
#         output = input + output

#     return output.flatten(2).transpose(1,2)


# def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
#     assert gamma <= 1/4
#     B, N, C = input.shape
#     if patch_resolution is None:
#         sqrt_N = int(N ** 0.5)
#         if sqrt_N * sqrt_N == N:
#             patch_resolution = (sqrt_N, sqrt_N)
#         else:
#             raise ValueError("Cannot infer a valid patch_resolution for the given input shape.")

#     input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
#     input = input
#     B, C, H, W = input.shape
#     output = torch.zeros_like(input)
#     output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
#     output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
#     output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
#     output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
#     output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
#     return output.flatten(2).transpose(1, 2)


## method 1 q_shift in fact is  omnishift
# def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
#     assert gamma <= 1/4
#     B, N, C = input.shape
#     if patch_resolution is None:
#         sqrt_N = int(N ** 0.5)
#         if sqrt_N * sqrt_N == N:
#             patch_resolution = (sqrt_N, sqrt_N)
#         else:
#             raise ValueError("Cannot infer a valid patch_resolution for the given input shape.")

#     input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
#     B, C, H, W = input.shape
    
#     # 创建输出张量
#     output = torch.zeros_like(input)
    
#     # 定义周边八个像素的权重
#     weights = torch.tensor([
#         [1/20, 1/5, 1/20],
#         [1/5, 0, 1/5],
#         [1/20, 1/5, 1/20]
#     ], dtype=input.dtype, device=input.device)
    
#     # 对每个像素点的周边八个像素进行加权平均
#     for i in range(shift_pixel, H - shift_pixel):
#         for j in range(shift_pixel, W - shift_pixel):
#             # 提取周边八个像素
#             patch = input[:, :, i-shift_pixel:i+shift_pixel+1, j-shift_pixel:j+shift_pixel+1]
#             # 加权平均
#             weighted_patch = torch.einsum('bchw,hw->bc', patch, weights)
#             # 将加权平均结果赋值给输出张量
#             output[:, :, i, j] = weighted_patch
    
#     # 处理边界情况
#     output[:, :, :shift_pixel, :] = input[:, :, :shift_pixel, :]
#     output[:, :, H-shift_pixel:, :] = input[:, :, H-shift_pixel:, :]
#     output[:, :, :, :shift_pixel] = input[:, :, :, :shift_pixel]
#     output[:, :, :, W-shift_pixel:] = input[:, :, :, W-shift_pixel:]
    
#     # 返回结果
#     return output.flatten(2).transpose(1, 2)

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


class Block(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False, k_norm=True,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, init_mode,
                                    k_norm=k_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, hidden_rate,
                                    init_mode, k_norm=k_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    """
    Perform ZigZag scan on a 4D tensor (B, C, H, W).
    
    Args:
        input (torch.Tensor): Input tensor of shape (B, C, H, W).
    
    Returns:
        torch.Tensor: Flattened 1D sequence of the tensor elements in ZigZag order, with shape (B, C, H * W).
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
    output = torch.zeros((B, C, H * W), dtype=input.dtype, device=input.device)

    for b in range(B):
        for c in range(C):
            # Get the current 2D matrix for the batch and channel
            matrix = input[b, c]
            # Initialize the output list of lists
            zigzag_output = [[] for _ in range(2 * H - 1)]

            # First half: l in [0, H)
            for l in range(H):
                # Append elements in the order: [l, 0], [l-1, 1], [l-2, 2], ..., [0, l]
                for i in range(l + 1):
                    zigzag_output[l].append(matrix[l - i, i])
                # If l is odd, reverse the sequence
                if l % 2 == 1:
                    zigzag_output[l] = zigzag_output[l][::-1]

            # Second half: l in (H, 2*H-1)
            for l in range(H, 2 * H - 1):
                # Append elements in the order: [H-1, l + 1 - H], [H-2, l + 2 - H], ..., [l + 1 - H, H-1]
                for i in range(2 * H - 1 - l):
                    zigzag_output[l].append(matrix[H - 1 - i, l + i - (H - 1)])
                # If l is odd, reverse the sequence
                if l % 2 == 1:
                    zigzag_output[l] = zigzag_output[l][::-1]

            # Flatten the output to get the final ZigZag sequence
            zigzag_sequence = [item for sublist in zigzag_output for item in sublist]
            output[b,c] = torch.tensor(zigzag_sequence, dtype=input.dtype, device=input.device)

    return output.flatten(2).transpose(1,2)
def zigzag_scan_2BNC_reverse(input):
    """
    Perform ZigZag scan on a 4D tensor (B, C, H, W).
    
    Args:
        input (torch.Tensor): Input tensor of shape (B, C, H, W).
    
    Returns:
        torch.Tensor: Flattened 1D sequence of the tensor elements in ZigZag order, with shape (B, C, H * W).
    """
    B, C, H, W = input.shape
    output = torch.zeros((B, C, H * W), dtype=input.dtype, device=input.device)

    for b in range(B):
        for c in range(C):
            # Get the current 2D matrix for the batch and channel
            matrix = input[b, c]
            # Initialize the output list of lists
            zigzag_output = [[] for _ in range(2 * H - 1)]

            # First half: l in [0, H)
            for l in range(H):
                # Append elements in the order: [l, 0], [l-1, 1], [l-2, 2], ..., [0, l]
                for i in range(l + 1):
                    zigzag_output[l].append(matrix[l - i, i])
                # If l is odd, reverse the sequence
                if l % 2 == 1:
                    zigzag_output[l] = zigzag_output[l][::-1]

            # Second half: l in (H, 2*H-1)
            for l in range(H, 2 * H - 1):
                # Append elements in the order: [H-1, l + 1 - H], [H-2, l + 2 - H], ..., [l + 1 - H, H-1]
                for i in range(2 * H - 1 - l):
                    zigzag_output[l].append(matrix[H - 1 - i, l + i - (H - 1)])
                # If l is odd, reverse the sequence
                if l % 2 == 1:
                    zigzag_output[l] = zigzag_output[l][::-1]

            # Flatten the output to get the final ZigZag sequence
            zigzag_sequence = [item for sublist in zigzag_output for item in sublist]
            output[b,c] = torch.tensor(zigzag_sequence, dtype=input.dtype, device=input.device)

    return output.transpose(1,2)

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
            # 这里改动一下 
            # x = zigzag_scan_2BNC_reverse(x)
        x = x + self.drop_path(self.spatial_mix(self.ln1(x), patch_resolution))

        # Channel Mixing (FFN-like)
        x = x + self.drop_path(self.channel_mix(self.ln2(x), patch_resolution))
        
        if len(x.shape) == 3:
            out = x.transpose(1, 2).reshape(B, C, H, W)
        return out



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super().__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, k, groups=ch_in, padding=k // 2),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, 1),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, 1),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for _ in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
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


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch_in, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, 3, padding=1, groups=2),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, 1),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, 1),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        return self.conv(x)


class CMUNeXt_rwkv_zigzagBNC_1_3_128_256_768(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, 
                 dims=[16, 32, 128, 256, 768], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7], 
                 shift_pixel = 1,
                 BNC='zigzag',
                 args_vit=None):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(2, 2)
        self.stem = conv_block(input_channel, dims[0])
        self.encoder1 = CMUNeXtBlock(dims[0], dims[0], depths[0], kernels[0])
        self.encoder2 = CMUNeXtBlock(dims[0], dims[1], depths[1], kernels[1])
        self.encoder3 = CMUNeXtBlock(dims[1], dims[2], depths[2], kernels[2])
        self.encoder4 = CMUNeXtBlock(dims[2], dims[3], depths[3], kernels[3])
        self.encoder5 = CMUNeXtBlock(dims[3], dims[4], depths[4], kernels[4])

        # PATCH_SIZE = args_vit.patch_size if args_vit else 1
        # NUM_PATCHES = (16 // PATCH_SIZE) ** 2
        # IN_CHANNELS = dims[4]
        # EMBED_DIM = dims[4]

        self.rwkv_bottleneck = VRWKV_Bottleneck(
        n_embd=dims[4],
        n_layer=12,
        layer_id=0,
        shift_mode='q_shift',
        channel_gamma=1/4,
        shift_pixel=1,
        hidden_rate=4,
        init_mode='fancy',
        drop_path=0.1,
        k_norm=True
    )

        self.Up5 = up_conv(dims[4], dims[3])
        self.Up_conv5 = fusion_conv(dims[3] * 2, dims[3])
        self.Up4 = up_conv(dims[3], dims[2])
        self.Up_conv4 = fusion_conv(dims[2] * 2, dims[2])
        self.Up3 = up_conv(dims[2], dims[1])
        self.Up_conv3 = fusion_conv(dims[1] * 2, dims[1])
        self.Up2 = up_conv(dims[1], dims[0])
        self.Up_conv2 = fusion_conv(dims[0] * 2, dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, 1)

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

        # x5_ = x5.
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



def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# if __name__ == "__main__":
#     # 假设 model 是动态创建的
#     from torchstat import stat
    
#     model = CMUNeXt_rwkv_1_3_128_256_768(dims=[16, 32, 128, 256, 384])  # 这里可以替换为其他模型
#     stat(model, (3,256, 256))

# Example usage
if __name__ == "__main__":
    # Input shape: (B, N, C)
    B, N, C = 2, 196, 768
    x = torch.randn(B, N, C)
    # x = torch.randn(B, 768, 16, 16)
    input = torch.randn(B, 3, 256, 256)
    x = input.cuda()
    # Bottleneck module
    bottleneck = VRWKV_Bottleneck(
        n_embd=768,
        n_layer=12,
        layer_id=0,
        shift_mode='q_shift',
        channel_gamma=1/4,
        shift_pixel=1,
        hidden_rate=4,
        init_mode='fancy',
        drop_path=0.1,
        k_norm=True
    )
    
    
    # bottleneck = bottleneck.cuda()
    # # Forward pass
    model = CMUNeXt_rwkv_zigzagBNC_1_3_128_256_768().cuda()
    # output = model(x)
    print((count_params(model)))

    output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    
    
#  class VRWKV(BaseBackbone):
#     def __init__(self,
#                  img_size=224,
#                  patch_size=16,
#                  in_channels=3,
#                  out_indices=-1,
#                  drop_rate=0.,
#                  embed_dims=256,
#                  depth=12,
#                  drop_path_rate=0.,
#                  channel_gamma=1/4,
#                  shift_pixel=1,
#                  shift_mode='q_shift',
#                  init_mode='fancy',
#                  post_norm=False,
#                  k_norm=True,
#                  init_values=None,
#                  hidden_rate=4,
#                  final_norm=True,
#                  interpolate_mode='bicubic',
#                  with_cp=False):
#         super().__init__()
#         self.embed_dims = embed_dims
#         self.num_extra_tokens = 0
#         self.num_layers = depth
#         self.drop_path_rate = drop_path_rate

#         self.patch_embed = PatchEmbed(
#             in_channels=in_channels,
#             input_size=img_size,
#             embed_dims=self.embed_dims,
#             conv_type='Conv2d',
#             kernel_size=patch_size,
#             stride=patch_size,
#             bias=True)
        
#         self.patch_resolution = self.patch_embed.init_out_size
#         num_patches = self.patch_resolution[0] * self.patch_resolution[1]

#         # Set position embedding
#         self.interpolate_mode = interpolate_mode
#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, num_patches, self.embed_dims))
        
#         self.drop_after_pos = nn.Dropout(p=drop_rate)

#         if isinstance(out_indices, int):
#             out_indices = [out_indices]
#         assert isinstance(out_indices, Sequence), \
#             f'"out_indices" must by a sequence or int, ' \
#             f'get {type(out_indices)} instead.'
#         for i, index in enumerate(out_indices):
#             if index < 0:
#                 out_indices[i] = self.num_layers + index
#             assert 0 <= out_indices[i] <= self.num_layers, \
#                 f'Invalid out_indices {index}'
#         self.out_indices = out_indices
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         self.layers = ModuleList()
#         for i in range(self.num_layers):
#             self.layers.append(Block(
#                 n_embd=embed_dims,
#                 n_layer=depth,
#                 layer_id=i,
#                 channel_gamma=channel_gamma,
#                 shift_pixel=shift_pixel,
#                 shift_mode=shift_mode,
#                 hidden_rate=hidden_rate,
#                 drop_path=dpr[i],
#                 init_mode=init_mode,
#                 post_norm=post_norm,
#                 k_norm=k_norm,
#                 init_values=init_values,
#                 with_cp=with_cp
#             ))

#         self.final_norm = final_norm
#         if final_norm:
#             self.ln1 = nn.LayerNorm(self.embed_dims)

#         # self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             if hasattr(m, 'scale_init'):
#                 if m.scale_init == 0:
#                     nn.init.zeros_(m.weight)
#                 elif m.scale_init == 1:
#                     nn.init.orthogonal_(m.weight)
#                 else:
#                     raise NotImplementedError
#             else:
#                 nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         B = x.shape[0]
#         x, patch_resolution = self.patch_embed(x)

#         x = x + resize_pos_embed(
#             self.pos_embed,
#             self.patch_resolution,
#             patch_resolution,
#             mode=self.interpolate_mode,
#             num_extra_tokens=self.num_extra_tokens)
        
#         x = self.drop_after_pos(x)

#         outs = []
#         for i, layer in enumerate(self.layers):
#             x = layer(x, patch_resolution)

#             if i == len(self.layers) - 1 and self.final_norm:
#                 x = self.ln1(x)

#             if i in self.out_indices:
#                 B, _, C = x.shape
#                 patch_token = x.reshape(B, *patch_resolution, C)
#                 patch_token = patch_token.permute(0, 3, 1, 2)

#                 out = patch_token
#                 outs.append(out)

#         return tuple(outs)