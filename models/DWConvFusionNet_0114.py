import torch
import torch.nn as nn
import math
from typing import Optional
import pywt
import torchvision.models as models
from timm.models.layers import DropPath
from einops import rearrange
import os,sys
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


def in_vertical_scan(input_tensor):
    """
    对输入张量进行垂直向内扫描
    """
    B, C, H, W = input_tensor.shape
    mid = H // 2
    # 将 NCHW 张量转置为 NHWC 格式，以便对倒数第二维（垂直方向）进行操作
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    top_half = input_tensor[:, :mid, :, :]
    bottom_half = torch.flip(input_tensor[:, mid:, :, :], dims=[1])
    # 拼接上下两部分
    # print("top_half",top_half.flatten())
    # print("bottom", bottom_half.flatten())    
    combined = torch.cat([top_half, bottom_half], dim=1)
    # 将张量转置为 (B * W, H, C) 形状
    transposed = rearrange(combined, 'B H W C -> (B W) H C').contiguous()
    return transposed.view(B, W * H, C) #.permute(0, 2, 1)

# def q_shift(input, shift_pixel=1, gamma=1 / 4, resolution=None, scan_type='horizontal_forward'):
#     """
#     对输入张量进行 q_shift 操作，并根据 scan_type 选择不同的扫描方式
#     """
#     assert gamma <= 1 / 4
#     if len(input.shape) == 3:
#         B, N, C = input.shape
#     N = input.shape[1]
#     if resolution is None:
#         sqrt_N = int(N ** 0.5)
#         if sqrt_N * sqrt_N == N:
#             resolution = (sqrt_N, sqrt_N)
#         else:
#             raise ValueError("Cannot infer a valid resolution for the given input shape.")
#     if len(input.shape) == 3:
#         input = input.transpose(1, 2).reshape(B, C, resolution[0], resolution[1])
#     # B, C, H, W = input.shape
#     # output = torch.zeros_like(input)

#     # if scan_type == 'horizontal_forward':
#     #     output = horizontal_forward_scan(input)
#     # elif scan_type == 'horizontal_backward':
#     #     output = horizontal_backward_scan(input)
#     # elif scan_type == 'vertical_forward':
#     #     output = vertical_forward_scan(input)
#     # elif scan_type == 'vertical_backward':
#     #     output = vertical_backward_scan(input)
#     # elif scan_type == 'in_horizontal':
#     #     output = in_horizontal_scan(input)
#     # elif scan_type == 'out_horizontal':
#     #     output = out_horizontal_scan(input)
#     # elif scan_type == 'in_vertical':
#     #     output = in_vertical_scan(input)
#     # elif scan_type == 'out_vertical':
#     #     output = out_vertical_scan(input)
#     # else:
#     #     raise ValueError(f"Invalid scan_type: {scan_type}")

#     # # output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]
#     # # output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[:, int(C * gamma):int(C * gamma * 2), :, shift_pixel:W]
#     # # output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[:, int(C * gamma * 2):int(C * gamma * 3), 0:H - shift_pixel, :]
#     # # output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[:, int(C * gamma * 3):int(C * gamma * 4), shift_pixel:H, :]
#     # # output[:, int(C * gamma * 4):,...] = input[:, int(C * gamma * 4):,...]
#     output = in_vertical_scan(input)
#     return output #.flatten(2).transpose(1, 2)

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

class RWKVBlock(nn.Module):
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
    def __init__(self, n_embd, n_layer, shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1, 
                 hidden_rate=4, init_mode='fancy', drop_path=0., key_norm=True, ffn_first=False, use_gc=True):
        super().__init__()
        self.use_gc = use_gc

        # 定义 GSC 模块
        if use_gc:
            self.gsc = GSC(n_embd)

        self.block_forward_rwkv = RWKVBlock(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=0,
            hidden_rate=4,
            init_mode=init_mode,
            key_norm=key_norm,
            ffn_first=ffn_first
        )
        self.block_backward_rwkv = RWKVBlock(
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

        # 使用 GSC
        if self.use_gc:
            z = self.gsc(z)

        z_f = z.flatten(2).transpose(1, 2)
        z_r = z.transpose(2, 3).flatten(2).transpose(1, 2)

        rwkv_f = self.block_forward_rwkv(z_f)
        rwkv_r = self.block_backward_rwkv(z_r)

        fused = rwkv_f + rwkv_r
        if len(fused.shape) == 3:
            fused = fused.transpose(1, 2).view(B, C, H, W)
        return fused

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

class BasicConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpsampleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpsampleConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
class SkipConnection(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ChannelFusionConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ChannelFusionConv, self).__init__()
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

class GSC(nn.Module):
    def __init__(self, in_channels):
        super(GSC, self).__init__()
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(in_channels)
        self.nonlinear = nn.ReLU()

        self.proj2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.nonlinear2 = nn.ReLU()

        self.proj3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.norm3 = nn.InstanceNorm2d(in_channels)
        self.nonlinear3 = nn.ReLU()

        self.proj4 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.norm4 = nn.InstanceNorm2d(in_channels)
        self.nonlinear4 = nn.ReLU()

    def forward(self, x):
        x_residual = x
        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonlinear(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonlinear2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonlinear3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonlinear4(x)

        return x + x_residual

class DWConvFusionBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3, use_gs=True):
        super(DWConvFusionBlock, self).__init__()
        self.gsc = GSC(ch_in)  # 添加 GSC 模块
        self.use_gs = use_gs 
        self.block = nn.Sequential(
            *[nn.Sequential(
                SkipConnection(nn.Sequential(
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
        self.up = BasicConvBlock(ch_in, ch_out)

    def forward(self, x):
        if self.use_gs:
            
            x = self.gsc(x)  # 应用 GSC
        x = self.block(x)
        # if self.use_gs:
        
        x = self.up(x)
        return x

class DWConvFusionNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24, 48, 96, 192,384], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7],use_gs=False,use_wt=False):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of DWConvFusion blocks
            kernels: kernel size of DWConvFusion blocks
        """
        super(DWConvFusionNet, self).__init__()
        # Encoder
        self.use_gs=use_gs
        self.use_wt = use_wt
        if self.use_wt:
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = BasicConvBlock(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = DWConvFusionBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0],use_gs=use_gs)
        self.encoder2 = DWConvFusionBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1],use_gs=use_gs)
        self.encoder3 = DWConvFusionBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2],use_gs=use_gs)
        self.encoder4 = DWConvFusionBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3],use_gs=use_gs)
        self.encoder5 = DWConvFusionBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4],use_gs=use_gs)
        # Decoder
        self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
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
    
    
class DWConvFusionNet_5(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24, 48, 96, 192,384], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7],use_gs=False,use_wt=False):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of DWConvFusion blocks
            kernels: kernel size of DWConvFusion blocks
        """
        super(DWConvFusionNet_5, self).__init__()
        # Encoder
        self.use_gs=use_gs
        self.use_wt = use_wt
        if self.use_wt:
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = BasicConvBlock(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = DWConvFusionBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0],use_gs=use_gs)
        self.encoder2 = DWConvFusionBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1],use_gs=use_gs)
        self.encoder3 = DWConvFusionBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2],use_gs=use_gs)
        self.encoder4 = DWConvFusionBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3],use_gs=use_gs)
        self.encoder5 = DWConvFusionBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4],use_gs=use_gs)
        # Decoder
        self.Up5 = UpsampleConv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = ChannelFusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = UpsampleConv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = ChannelFusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = UpsampleConv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = ChannelFusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = UpsampleConv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = ChannelFusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        
        self.Brwkv_4 = BinaryOrientatedRWKV2D(
            n_embd=dims[-2],
            n_layer=8, 
            shift_mode='q_shift',
            channel_gamma=1/4,
            shift_pixel=1,
            hidden_rate=4,
            init_mode="fancy",
            drop_path=0,
            key_norm=True,
            use_gc=use_gs  # 传递 use_gs 参数
        )
        self.Brwkv_5 = BinaryOrientatedRWKV2D(
            n_embd=dims[-1],
            n_layer=8, 
            shift_mode='q_shift',
            channel_gamma=1/4,
            shift_pixel=1,
            hidden_rate=4,
            init_mode="fancy",
            drop_path=0,
            key_norm=True,
            use_gc=use_gs  # 传递 use_gs 参数
        )
            
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
        
        x5 = self.Brwkv_5(x5)
        
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
    
    
def dwconvfusionnet(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
    return DWConvFusionNet(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def dwconvfusionnet_s(input_channel=3, num_classes=1, dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9]):
    return DWConvFusionNet(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def dwconvfusionnet_l(input_channel=3, num_classes=1, dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7]):
    return DWConvFusionNet(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)   
 
def dwconvfusionnet_rwkv(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
    return DWConvFusionNet_5(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def dwconvfusionnet_rwkv_s(input_channel=3, num_classes=1, dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9]):
    return DWConvFusionNet_5(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def dwconvfusionnet_rwkv_l(input_channel=3, num_classes=1, dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7]):
    return DWConvFusionNet_5(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)    
    


if __name__ == "__main__":
#     x = torch.rand((1, 3, 256, 256)).cuda()


    import os 
    os.environ['CUDA_VISIBLE_DEVICES']='7'

    import time 
    from thop import profile, clever_format
    
    '''
    !!!!!!!!
    Caution: Please comment out the code related to reparameterization and retain only the 5x5 convolutional layer in the OmniShift.
    !!!!!!!!
    '''
    x=torch.zeros((1, 3, 256, 256)).type(torch.FloatTensor).cuda()    
    # # model = LORDD_s_for_FLOPs()
    # model = dwconvfusionnet_s()
    # model.cuda()
    
    # since = time.time()
    # # y=model(x)
    # print("time", time.time()-since)
    
    # flops, params = profile(model, inputs=(x, ))  
    # flops, params = clever_format([flops, params], '%.2f')
    # print(f"FLOPs: {flops}")
    # print(f"Params: {params}")    

    # 定义模型列表
    models = {
        "dwconvfusionnet": dwconvfusionnet(),
        "dwconvfusionnet_s": dwconvfusionnet_s(),
        "dwconvfusionnet_l": dwconvfusionnet_l(),
        "dwconvfusionnet_rwkv": dwconvfusionnet_rwkv(),
        "dwconvfusionnet_rwkv_s": dwconvfusionnet_rwkv_s(),
        "dwconvfusionnet_rwkv_l": dwconvfusionnet_rwkv_l(),
    }

    # 遍历模型并计算 FLOPs 和 Params
    for name, model in models.items():
        model.cuda()  # 将模型移动到 GPU
        model.eval()  # 设置为评估模式

        # 预热 GPU
        with torch.no_grad():
            _ = model(x)

        # 计算 FLOPs 和 Params
        flops, params = profile(model, inputs=(x,))
        flops, params = clever_format([flops, params], '%.2f')

        # 打印结果
        print(f"Model: {name}")
        print(f"FLOPs: {flops}")
        print(f"Params: {params}")
        print("-" * 50)    