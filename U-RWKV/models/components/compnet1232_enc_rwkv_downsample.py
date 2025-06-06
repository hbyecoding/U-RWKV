import os,sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.layers import DropPath
# from torch_dwconv import DepthwiseConv2d
# from mmcv.runner.base_module import BaseModule, ModuleList
# from mmcls.models.utils import resize_pos_embed
from einops import rearrange


T_MAX = 4096 #128*128 2048 均不可以 # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load


wkv_cuda = load(name="wkv", sources=[os.path.join(os.getcwd(),'Tan9','src','network', 'conv_based','transformer_learning', 'cuda', 'wkv_op.cpp'), os.path.join(os.getcwd(),'Tan9','src','network', 'conv_based', 'transformer_learning', 'cuda', 'wkv_cuda.cu')],
                verbose=True, extra_cuda_cflags=['-res-usage','--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}']) #'--maxrregcount 60', 

# wkv_cuda = load(name="wkv", sources=["/data/hongboye/projects/Restore-RWKV/model/cuda/wkv_op.cpp", "/data/hongboye/projects/Restore-RWKV/model/cuda/wkv_cuda.cu"],
#                 verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])

def q_shift(input, shift_pixel=1, gamma=1/4, resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    if resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError("Cannot infer a valid resolution for the given input shape.")

    input = input.transpose(1, 2).reshape(B, C, resolution[0], resolution[1])
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

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

class SpatialInteractionMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, key_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd

        # 初始化 fancy 模式的权重
        with torch.no_grad():
            ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))  # 0 to 1
            ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0

            # fancy time_decay
            decay_speed = torch.ones(self.n_embd)
            for h in range(self.n_embd):
                decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.spatial_decay = nn.Parameter(decay_speed)

            # fancy time_first
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

            # fancy time_mix
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        # 设置 shift 相关参数
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        # 定义线性层
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        
        # 可选的 LayerNorm
        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        
        # 输出线性层
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        # 初始化线性层的 scale_init
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0
        self.value.scale_init = 1

    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, resolution)
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

    def forward(self, x, resolution=None):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution)
        rwkv = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
        rwkv = sr * rwkv
        rwkv = self.output(rwkv)
        return rwkv

class SpectralMixer(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, key_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd

        # 初始化 fancy 模式的权重
        with torch.no_grad():
            ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0
            x = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                x[0, 0, i] = i / self.n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        # 设置 shift 相关参数
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        # 定义线性层
        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        # 可选的 LayerNorm
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None

        # 初始化线性层的 scale_init
        self.value.scale_init = 0
        self.receptance.scale_init = 0
        self.key.scale_init = 1

    def forward(self, x, resolution=None):
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x

        # 计算 key、value 和 receptance
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

        # 计算最终输出
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


import torch
import torch.nn as nn
from einops import rearrange
# from .dwt import DWT_2D  # 假设 DWT_2D 已经实现
import pywt

# class DWT_2D(nn.Module):
#     def __init__(self, n_embd, wave='haar'):
#         super(DWT_2D, self).__init__()
#         self.n_embd = n_embd
#         self.groups = n_embd // 4 
#         w = pywt.Wavelet(wave)
#         dec_hi = torch.Tensor(w.dec_hi[::-1])  # 高通滤波器
#         dec_lo = torch.Tensor(w.dec_lo[::-1])  # 低通滤波器

#         # 构造四个滤波器核
#         w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)  # 低频子带
#         w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)  # 水平高频子带
#         w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)  # 垂直高频子带
#         w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  # 对角线高频子带

#         # 注册滤波器核为模型的缓冲区
#         self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
#         self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
#         self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
#         self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))
#         # self.conv_xx = nn.Conv2d(in_channels=n_embd,out_channels=self.groups, kernel_size=3,padding=1,groups=self.groups)

#     def forward(self, x):
#         dim = x.shape[1]  # 输入通道数
#         groups = dim // 4  # 每组输入通道数为 4

#         # 对输入通道进行分组卷积
#         x_ll = F.conv2d(x, self.w_ll.expand(groups, 1, 3, 3), stride=1, padding=1, groups=groups)
#         x_lh = F.conv2d(x, self.w_lh.expand(groups, 1, 3, 3), stride=1, padding=1, groups=groups)
#         x_hl = F.conv2d(x, self.w_hl.expand(groups, 1, 3, 3), stride=1, padding=1, groups=groups)
#         x_hh = F.conv2d(x, self.w_hh.expand(groups, 1, 3, 3), stride=1, padding=1, groups=groups)

#         # 将四个子带拼接在一起
#         x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
#         print("x_ll.shape", x_ll.shape, "x.shape", x.shape)
#         return x
    
class DWT_2D(nn.Module):
    def __init__(self, wave='haar'): 
        #,ch_in = None self.dim = ch_in
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])  # 高通滤波器
        dec_lo = torch.Tensor(w.dec_lo[::-1])  # 低通滤波器

        # 构造四个滤波器核
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)  # 低频子带
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)  # 水平高频子带
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)  # 垂直高频子带
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  # 对角线高频子带

       
        # 注册滤波器核为模型的缓冲区
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # if self.dim is None:
        #     self.
        dim = x.shape[1]  # 输入通道数
        groups = int(dim // 4)  # 每组输入通道数为 4

        # 对输入通道进行分组卷积
        x_ll = torch.nn.functional.conv2d(x, self.w_ll.expand(groups, 4, -1, -1), stride=2, groups=groups)
        x_lh = torch.nn.functional.conv2d(x, self.w_lh.expand(groups, 4, -1, -1), stride=2, groups=groups)
        x_hl = torch.nn.functional.conv2d(x, self.w_hl.expand(groups, 4, -1, -1), stride=2, groups=groups)
        x_hh = torch.nn.functional.conv2d(x, self.w_hh.expand(groups, 4, -1, -1), stride=2, groups=groups)

        # 将四个子带拼接在一起
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # print("x_ll.shape",x_ll.shape,"x.shape",x.shape)
        return x
    



# class FreqInteractionMixer(nn.Module):
#     def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy',
#                  key_norm=False, wave='haar'):
#         super().__init__()
#         self.layer_id = layer_id
#         self.n_layer = n_layer
#         self.n_embd = n_embd

#         # 初始化 DWT_2D 模块
#         self.dwt = DWT_2D(n_embd=n_embd,wave=wave)

#         # 定义线性层
#         hidden_sz = int(hidden_rate * n_embd)
#         # self.key = nn.Linear(n_embd * 4, hidden_sz, bias=False)  # 输入通道数变为 4 倍
#         # self.receptance = nn.Linear(n_embd * 4, n_embd, bias=False)  # 输入通道数变为 4 倍
#         # self.value = nn.Linear(hidden_sz, n_embd, bias=False)

#         self.key = nn.Linear(n_embd, hidden_sz, bias=False)
#         self.receptance = nn.Linear(n_embd, n_embd, bias=False)
#         self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        
#         # 可选的 LayerNorm
#         if key_norm:
#             self.key_norm = nn.LayerNorm(hidden_sz)
#         else:
#             self.key_norm = None

#     def forward(self, x, resolution=None):
#         # print("x shape",x.shape)
#         if resolution:
#             h, w = resolution
#         else:
#             L= x.shape[1]
#             h=w = int(math.sqrt(int(L)))
#         # 将输入特征图从 (B, L, C) 转换为 (B, C, H, W)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

#         # 使用 DWT_2D 进行多尺度分解
#         x = self.dwt(x)  # 输出形状: (B, C, H/2, W/2)

#         # 将特征图从 (B, C, H/2, W/2) 转换为 (B, L', C) 
#         x = rearrange(x, 'b c h w -> b (h w) c')

#         # 通道混合操作
#         k = self.key(x)  # 线性变换
#         k = torch.square(torch.relu(k))  # 非线性激活
#         if self.key_norm is not None:
#             k = self.key_norm(k)  # 归一化
#         kv = self.value(k)  # 线性变换

#         # 计算最终输出
#         print("kv.shape",kv.shape)
#         x = kv * torch.sigmoid(self.receptance(x)) 

#         return x

# class LoRABlock(nn.Module):
#     def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
#                  init_mode='fancy', key_norm=False,ffn_first=False):
#         super().__init__()
#         self.layer_id = layer_id
#         self.depth = depth  # 新增 depth 参数
#         self.ffn_first = ffn_first    
#         # LayerNorm 层
#         self.ln1 = nn.LayerNorm(n_embd)
#         self.ln2 = nn.LayerNorm(n_embd)

#         # 空间混合模块
        
#         self.att = SpatialInteractionMix(
#             n_embd=n_embd,
#             n_layer=n_layer,
#             layer_id=layer_id,
#             shift_mode='q_shift',
#             key_norm=key_norm
#         )
        
        # # 频域混合模块
        # self.ffn = SpectralMixer(
        #     n_embd=n_embd,
        #     n_layer=n_layer,
        #     layer_id=layer_id,
        #     shift_mode='q_shift',
        #     key_norm=key_norm
        # )
#         # 可学习的缩放参数
#         self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
#         self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

#     def forward(self, x):
#         # print(x.shape)
#         if len(x.shape) == 4:
#             b,c,h,w = x.shape
#             x = rearrange(x, 'b c h w -> b (h w) c')
#         b, n, c= x.shape  #, 
#         h = w = int(n ** 0.5) 
#         resolution = (h, w)

#         # 空间混合
#         if self.ffn_first:
#             x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
#             x = x + self.gamma1 * self.att(self.ln1(x), resolution)
#             # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

#             # # 频域混合
#             # x = rearrange(x, 'b c h w -> b (h w) c')
            
#         else:
#             x = x + self.gamma1 * self.att(self.ln1(x), resolution)
#             # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

#             # # 频域混合
#             # x = rearrange(x, 'b c h w -> b (h w) c')
#             x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)            
#         x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

#         return x

class LoRABlock(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, depth=1, hidden_rate=4,
                 init_mode='fancy', key_norm=False,ffn_first=False):
        super().__init__()
        self.layer_id = layer_id
        self.depth = depth  # 新增 depth 参数
        self.ffn_first = ffn_first    
        # LayerNorm 层
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # 空间混合模块
        
        self.att = SpatialInteractionMix(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            shift_mode='q_shift',
            key_norm=key_norm
        )
        
        # 频域混合模块
        self.ffn = SpectralMixer(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            shift_mode='q_shift',
            key_norm=key_norm
        )
        
        # 可学习的缩放参数
        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x):
        # print(x.shape)
        if len(x.shape) == 4:
            b,c,h,w = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
        b, n, c= x.shape  #, 
        h = w = int(n ** 0.5) 
        resolution = (h, w)

        # 空间混合
        if self.ffn_first:
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
            x = x + self.gamma1 * self.att(self.ln1(x), resolution)
            # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

            # # 频域混合
            # x = rearrange(x, 'b c h w -> b (h w) c')
            
        else:
            x = x + self.gamma1 * self.att(self.ln1(x), resolution)
            # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

            # # 频域混合
            # x = rearrange(x, 'b c h w -> b (h w) c')
            
            x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)            
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x

class BinaryOrientatedRWKV2D(nn.Module):
    def __init__(self, n_embd, n_layer, shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy', drop_path=0., key_norm=True,ffn_first=False):
        super().__init__()
        self.block_forward_rwkv = LoRABlock(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=0,
            hidden_rate=4,
            init_mode=init_mode,
            key_norm=key_norm,
            ffn_first=ffn_first
        )
        self.block_backward_rwkv = LoRABlock(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=1,
            hidden_rate=4,
            init_mode=init_mode,
            key_norm=key_norm,
            ffn_first=ffn_first
        )

    def forward(self, z):
        B, C, H, W = z.shape
        # 这里需要搞定一下
        z_f = z.flatten(2).transpose(1, 2)
        z_r = z.transpose(2, 3).flatten(2).transpose(1, 2)

        # rwkv_f = self.rwkv_forward(z_f)
        # rwkv_r = self.rwkv_reverse(z_r)
        rwkv_f = self.block_forward_rwkv(z_f)
        rwkv_r = self.block_backward_rwkv(z_r)
        
        fused = rwkv_f + rwkv_r
        if len(fused.shape) == 3:
            fused = fused.transpose(1, 2).view(B, C, H, W)
        return fused


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SELayer(out_c)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class LoRDBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super().__init__()
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
        self.up = ConvBlock(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x

class LoRDBlockEnc(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super().__init__()
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
        self.up = ConvBlock(ch_in, ch_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        p = self.pool(x)
        return x,p
    

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self, x):
        x = self.up(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool = DWT_2D(wave='haar') #,ch_in=in_channels

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DecoderBlock, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x, skip):
#         x = self.up(x)
#         x = torch.cat([x, skip], dim=1)
#         x = self.conv(x)
#         return x

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(in_c+out_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, x, s):
        x = self.upsample(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        x = self.r2(x)

        return x

class LoRA(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[32, 64, 128, 256,768], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(LoRA, self).__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        
        self.Brwkv = BinaryOrientatedRWKV2D(n_embd=self.n_emb,
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        x5 = self.Brwkv(x5)
        p5 = self.Brwkv(p5)
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1, out2

# def lord_without(is_Brwkv=False):
#     mdl = LoRA(dims=[24, 48, 96, 192, 384], depths=[1, 1, 1, 3, 1])
#     if is_Brwkv==False:
#         mdl.Brwkv = nn.Identity()
        
#     return mdl
# def lord_l(dims=[48, 96, 192, 384 , 768], depths=[12, 12, 12, 3, 1]):
#     model = LoRA(dims=dims, depths=depths)
#     return LoRA
    
class LoRA__5(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(LoRA__5, self).__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=self.n_emb, 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        # self.Brwkv = Block()

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        # x4 = self.Brwkv_4(x4)
        # p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        x5 = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)
        # print("x5.shape",x5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1 #, out2


class LoRA__5_Classification(nn.Module):
    def __init__(self, input_channel=3, num_classes=10, dims=[24, 48, 96, 192, 384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(LoRA__5_Classification, self).__init__()

        """ Shared Encoder """
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])

        """ RWKV Blocks """
        self.Brwkv_4 = BinaryOrientatedRWKV2D(
            n_embd=dims[3], 
            n_layer=12, 
            shift_mode='q_shift',
            channel_gamma=1/4,
            shift_pixel=1,
            hidden_rate=4,
            init_mode="fancy",
            drop_path=0,
            key_norm=True
        )
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(
            n_embd=dims[4], 
            n_layer=12, 
            shift_mode='q_shift',
            channel_gamma=1/4,
            shift_pixel=1,
            hidden_rate=4,
            init_mode="fancy",
            drop_path=0,
            key_norm=True
        )

        """ Global Average Pooling """
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化

        """ Output Layer """
        self.fc = nn.Linear(dims[4], num_classes)  # 全连接层，输出类别概率

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x5, p5 = self.e5(p4)

        """ RWKV Blocks """
        x5 = self.Brwkv_5(x5)

        """ Global Average Pooling """
        x5_pooled = self.global_pool(x5)  # 全局平均池化
        x5_pooled = x5_pooled.view(x5_pooled.size(0), -1)  # 展平

        """ Output """
        out = self.fc(x5_pooled)  # 全连接层，输出类别概率
        return out


    
class LoRA_4_5(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(LoRA_4_5, self).__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=self.n_emb, 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        # self.Brwkv = Block()

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x4 = self.Brwkv_4(x4)
        p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        # print("x5.shape", x5.shape)
        x5 = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1 #, out2

class LoRA_4_5_rwkv_ffn_first(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super().__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True,
                                            ffn_first=True)
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=self.n_emb, 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True,
                                            ffn_first=True)
        
        # self.Brwkv = Block()

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x4 = self.Brwkv_4(x4)
        p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        # print("x5.shape", x5.shape)
        x5 = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)
        
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1

class LoRA_3_4_5(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super().__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        # self.n_emb = dims[-1]
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)
        self.Brwkv_3 = BinaryOrientatedRWKV2D(n_embd=dims[2], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        self.Brwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[3], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        self.Brwkv_5 = BinaryOrientatedRWKV2D(n_embd=dims[-1], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            key_norm=True)
        
        # self.Brwkv = Block()

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x3 = self.Brwkv_3(x3)
        p3 = self.Brwkv_3(p3)
        x4, p4 = self.e4(p3)
        x4 = self.Brwkv_4(x4)
        p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        # x5 = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        # print("x5.shape", x5.shape)
        x5 = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1 #, out2



if __name__ == "__main__":
#     x = torch.rand((1, 3, 256, 256)).cuda()
#     model = LoRA_4_5().cuda()
#     y1, y2 = model(x)
#     print(y1.shape, y2.shape)
#     print(count_params(model))
#     print("END")

    import os 
    os.environ['CUDA_VISIBLE_DEVICES']='7'

    import time 
    from thop import profile, clever_format
    
    '''
    !!!!!!!!
    Caution: Please comment out the code related to reparameterization and retain only the 5x5 convolutional layer in the OmniShift.
    !!!!!!!!
    '''
    
    
    # x=torch.zeros((1, 384, 16, 16)).type(torch.FloatTensor).cuda()

    
    # x = torch.zeros((1, 256, 384)).cuda()
    # n_embd = 384
    # self_gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True).cuda()
    # self_ffn = FreqInteractionMixer(n_embd=384,
    #                          n_layer=12, layer_id=0).cuda()
    # model = FreqInteractionMixer(n_embd=384,
    #                          n_layer=12, layer_id=0).cuda()
    # self_ln2 = nn.LayerNorm(384).cuda()
    # y = self_gamma2 * self_ffn(self_ln2(x),(16,16))
    # print("y.shape",y.shape)
    
    # model = BinaryOrientatedRWKV2D(n_embd=384, n_layer=12)
    
    # x=torch.zeros((1, 768,32,32)).type(torch.FloatTensor).cuda()
    # model = BinaryOrientatedRWKV2D(n_embd=768, n_layer=12)
    x=torch.zeros((1, 3, 256, 256)).type(torch.FloatTensor).cuda()    
    model = LoRA__5()
    model.cuda() 
    
    since = time.time()
    # y=model(x)
    print("time", time.time()-since)
    
    flops, params = profile(model, inputs=(x, ))  
    flops, params = clever_format([flops, params], '%.6f')
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")