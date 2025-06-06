import torch
import torch.nn as nn

class ConvBlockChinBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, padding):
        super(ConvBlockChinBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.InstanceNorm2d(in_channels)
        self.nonlinear = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlinear(x)
        return x

class GSC_no_pm(nn.Module):
    def __init__(self, in_channels):
        super(GSC_no_pm, self).__init__()
        # 使用 ConvBlockChinBlock 简化重复的卷积、归一化和激活操作
        self.block1 = ConvBlockChinBlock(in_channels, kernel_size=3, padding=1)
        self.block2 = ConvBlockChinBlock(in_channels, kernel_size=3, padding=1)
        self.block3 = ConvBlockChinBlock(in_channels, kernel_size=1, padding=0)
        self.block4 = ConvBlockChinBlock(in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x_residual = x  # 保存残差连接

        # 第一组卷积操作
        x1 = self.block1(x)
        x1 = self.block2(x1)

        # 第二组卷积操作
        x2 = self.block3(x)

        # 融合结果
        x = x1 + x2  # 没有点乘操作

        # 最后一组卷积操作
        x = self.block4(x)

        # 残差连接
        return x + x_residual


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

# def q_shift(input, shift_pixel=1, gamma=1/4, resolution=None):
#     print("q_shift is called! OUT") 
#     assert gamma <= 1/4
#     # B, N, C = input.shape
#     # if resolution is None:
#     #     sqrt_N = int(N ** 0.5)
#     #     if sqrt_N * sqrt_N == N:
#     #         resolution = (sqrt_N, sqrt_N)
#     #     else:
#     #         raise ValueError("Cannot infer a valid resolution for the given input shape.")

#     # input = input.transpose(1, 2).reshape(B, C, resolution[0], resolution[1])
#     # input = input
#     # B, C, H, W = input.shape
#     # output = torch.zeros_like(input)
#     # output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
#     # output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
#     # output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
#     # output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
#     # output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
#     # return output.flatten(2).transpose(1, 2)
#     """
#     垂直向内扫描的 q_shift 实现。
#     """
#     B, N, C = input.shape
#     if resolution is None:
#         sqrt_N = int(N ** 0.5)
#         if sqrt_N * sqrt_N == N:
#             resolution = (sqrt_N, sqrt_N)
#         else:
#             raise ValueError("Cannot infer a valid resolution for the given input shape.")

#     input = input.transpose(1, 2).reshape(B, C, resolution[0], resolution[1])
#     B, C, H, W = input.shape
#     mid = H // 2
#     input = input.permute(0, 2, 3, 1)  # NHWC
#     top_half = input[:, :mid, :, :]
#     bottom_half = torch.flip(input[:, mid:, :, :], dims=[1])
#     combined = torch.cat([top_half, bottom_half], dim=1)
#     transposed = rearrange(combined, 'B H W C -> (B W) H C').contiguous()
#     return transposed.view(B, W * H, C)

def q_shift(input, shift_pixel=1, gamma=1/4, resolution=None):
    print("q_shift is called! OUT") 
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
    """
    垂直向内扫描的 q_shift 实现。
    """
    B, N, C = input.shape
    if resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError("Cannot infer a valid resolution for the given input shape.")

    input = input.transpose(1, 2).reshape(B, C, resolution[0], resolution[1])
    B, C, H, W = input.shape
    mid = H // 2
    input = input.permute(0, 2, 3, 1)  # NHWC
    top_half = input[:, :mid, :, :]
    bottom_half = torch.flip(input[:, mid:, :, :], dims=[1])
    combined = torch.cat([top_half, bottom_half], dim=1)
    transposed = rearrange(combined, 'B H W C -> (B W) H C').contiguous()
    return transposed.view(B, W * H, C)

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
            # print("U SpatialInteractionMix Out")

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
            # print("U SpectralMixer out")

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
    

class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1, drop_path=0., hidden_rate=4, init_mode='fancy', k_norm=True):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.att = SpatialInteractionMix(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, init_mode, k_norm)
        self.ffn = SpectralMixer(n_embd, n_layer, layer_id, shift_mode, channel_gamma, shift_pixel, hidden_rate, init_mode, k_norm)
        self.se = SELayer(n_embd)  # 插入 SELayer

    def apply_selayer(self, x, patch_resolution):
        """
        将 [B, N, C] 转换为 [B, C, H, W]，应用 SELayer，再转换回 [B, N, C]
        """
        B, N, C = x.shape
        H, W = patch_resolution
        # 转换为 [B, C, H, W]
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        # 应用 SELayer
        x = self.se(x)
        # 转换回 [B, N, C]
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x

    def forward(self, x, patch_resolution=None):
        # 第一个分支：Spatial Mix + SELayer
        x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
        x = self.apply_selayer(x, patch_resolution)  # 应用 SELayer
        # 第二个分支：Channel Mix
        x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
        return x




class BinaryOrientatedRWKV2D(nn.Module):
    def __init__(self, n_embd, n_layer, shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1, 
                 hidden_rate=4, init_mode='fancy', drop_path=0., key_norm=True, ffn_first=False, use_gc=False):
        super().__init__()
        self.use_gc = use_gc

        # # 定义 GSC 模块
        # if use_gc:
        #     self.gsc = GSC(n_embd)

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

        # # 使用 GSC
        # if self.use_gc:
        #     z = self.gsc(z)

        z_f = z.flatten(2).transpose(1, 2)
        z_r = z.transpose(2, 3).flatten(2).transpose(1, 2)

        rwkv_f = self.block_forward_rwkv(z_f)
        rwkv_r = self.block_backward_rwkv(z_r)

        fused = rwkv_f + rwkv_r
        if len(fused.shape) == 3:
            fused = fused.transpose(1, 2).view(B, C, H, W)
        return fused
    
class VRWKV(nn.Module):
    def __init(self, n_embd, n_layer=8,shift_mode='q_shift', init_mode='fancy', channel_gamma=1/4, shift_pixel=1, drop_path_rate=0., hidden_rate=4, k_norm=True):
        super().__init__()
        self.embed_dim
                
    def forward(self, z):
        B, C, H, W = z.shape

        # # 使用 GSC
        # if self.use_gc:
        #     z = self.gsc(z)

        z_f = z.flatten(2).transpose(1, 2)
        z_r = z.transpose(2, 3).flatten(2).transpose(1, 2)

        rwkv_f = self.block_forward_rwkv(z_f)
        rwkv_r = self.block_backward_rwkv(z_r)

        fused = rwkv_f + rwkv_r
        if len(fused.shape) == 3:
            fused = fused.transpose(1, 2).view(B, C, H, W)
        return fused





class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


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
    
# wo chin 4 with SE(4)
# class CMUNeXtBlock(nn.Module):
#     def __init__(self, ch_in, ch_out, depth=1, k=3):
#         super(CMUNeXtBlock, self).__init__()
#         self.block = nn.Sequential(
#             *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     # deep wise
#                     nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
#                     nn.GELU(),
#                     nn.BatchNorm2d(ch_in)
#                 )),

#             ) for i in range(depth)]
#         )
#         self.up = conv_block(ch_in, ch_out)
#         self.se = SELayer(ch_out, reduction=4)

#     def forward(self, x):
#         x = self.block(x)
#         x = self.up(x)
#         x = self.se(x)
#         return x


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
            nn.Upsample(scale_factor=2, mode='nearest'),
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


class CMUNeXt(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(CMUNeXt, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
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

class GSC(nn.Module):
    def __init__(self, in_channels):
        super(GSC, self).__init__()
        self.proj3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.proj1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.norm = nn.InstanceNorm2d(in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_residual = x
        x1 = self.proj3x3(x)
        x1 = self.norm(x1)
        x1 = self.relu(x1)
        x1g = self.norm(self.relu(self.proj1x1(x)))
        x1 = x1 * x1g  # 点乘操作
        x2 = self.norm(self.bn(self.proj3x3(x1)))
        return x2 + x_residual
    
    
class GSC_no_pm(nn.Module):
    def __init__(self, in_channels):
        super(GSC_no_pm, self).__init__()
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

        x = x1 + x2  # 没有点乘操作
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonlinear4(x)

        return x + x_residual

class CMUNeXt_GSC(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        super(CMUNeXt_GSC, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        
        # 在 encoder 之前添加 GSC
        self.gsc1 = GSC(dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        
        self.gsc2 = GSC(dims[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        
        self.gsc3 = GSC(dims[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        
        self.gsc4 = GSC(dims[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        
        self.gsc5 = GSC(dims[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        
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
        x1 = self.gsc1(x1)  # 添加 GSC
        x1 = self.encoder1(x1)
        
        x2 = self.Maxpool(x1)
        x2 = self.gsc2(x2)  # 添加 GSC
        x2 = self.encoder2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.gsc3(x3)  # 添加 GSC
        x3 = self.encoder3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.gsc4(x4)  # 添加 GSC
        x4 = self.encoder4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.gsc5(x5)  # 添加 GSC
        x5 = self.encoder5(x5)

        # Decoder 部分保持不变
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
    

class CMUNeXt_GSC_no_pm(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        super(CMUNeXt_GSC_no_pm, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        
        # 在 encoder 之前添加 GSC_no_pm
        self.gsc1 = GSC_no_pm(dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        
        self.gsc2 = GSC_no_pm(dims[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        
        self.gsc3 = GSC_no_pm(dims[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        
        self.gsc4 = GSC_no_pm(dims[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        
        self.gsc5 = GSC_no_pm(dims[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        
        # Decoder 部分保持不变
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
        x1 = self.gsc1(x1)  # 添加 GSC_no_pm
        x1 = self.encoder1(x1)
        
        x2 = self.Maxpool(x1)
        x2 = self.gsc2(x2)  # 添加 GSC_no_pm
        x2 = self.encoder2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.gsc3(x3)  # 添加 GSC_no_pm
        x3 = self.encoder3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.gsc4(x4)  # 添加 GSC_no_pm
        x4 = self.encoder4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.gsc5(x5)  # 添加 GSC_no_pm
        x5 = self.encoder5(x5)

        # Decoder 部分保持不变
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

def cmunext(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def cmunext_s(input_channel=3, num_classes=1, dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def cmunext_l(input_channel=3, num_classes=1, dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def cmunext_gsc(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt_GSC(input_channel=input_channel, num_classes=num_classes, dims=dims, depths=depths, kernels=kernels)

def cmunext_gsc_no_pm(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt_GSC_no_pm(input_channel=input_channel, num_classes=num_classes, dims=dims, depths=depths, kernels=kernels)


def cmunext_no_chin_4():
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
                    # nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                    # nn.GELU(),
                    # nn.BatchNorm2d(ch_in * 4),
                    # nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                    # nn.GELU(),
                    # nn.BatchNorm2d(ch_in)
                ) for i in range(depth)]
            )
            self.up = conv_block(ch_in, ch_out)

        def forward(self, x):
            x = self.block(x)
            x = self.up(x)
            return x
        
    class CMUNeXt(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
            """
            Args:
                input_channel : input channel.
                num_classes: output channel.
                dims: length of channels
                depths: length of cmunext blocks
                kernels: kernal size of cmunext blocks
            """
            super(CMUNeXt, self).__init__()
            # Encoder
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
            self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
            self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
            self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
            self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
            self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
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
    
    return CMUNeXt()

    
def cmunext_no_chin_4_withSE(reduction = 4):
    class SELayer(nn.Module):
        def __init__(self, channel, reduction=4):
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
                ) for i in range(depth)]
            )
            self.up = conv_block(ch_in, ch_out)
            self.se = SELayer(ch_out, reduction=reduction) # here hbye #TODO

        def forward(self, x):
            x = self.block(x)
            x = self.up(x)
            x = self.se(x)
            return x
        
    class CMUNeXt(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
            """
            Args:
                input_channel : input channel.
                num_classes: output channel.
                dims: length of channels
                depths: length of cmunext blocks
                kernels: kernal size of cmunext blocks
            """
            super(CMUNeXt, self).__init__()
            # Encoder
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
            self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
            self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
            self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
            self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
            self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
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
    
    return CMUNeXt()

def cmunext_no_chout_4_fusion():
    class fusion_conv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(fusion_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1)),
                # nn.GELU(),
                # nn.BatchNorm2d(ch_out * 4),
                # nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x
        
    class CMUNeXt(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
            """
            Args:
                input_channel : input channel.
                num_classes: output channel.
                dims: length of channels
                depths: length of cmunext blocks
                kernels: kernal size of cmunext blocks
            """
            super(CMUNeXt, self).__init__()
            # Encoder
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
            self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
            self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
            self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
            self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
            self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
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
    
    return CMUNeXt()

def cmunext_chin_4_no_fusion_dec_se_rwkv5(reduction = 4):
    class fusion_conv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(fusion_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1)),
                # nn.GELU(),
                # nn.BatchNorm2d(ch_out * 4),
                # nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )
            self.se = SELayer(ch_out, reduction=reduction)
        def forward(self, x):
            x = self.conv(x)
            x = self.se(x)
            return x

    class CMUNeXt(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
            """
            Args:
                input_channel : input channel.
                num_classes: output channel.
                dims: length of channels
                depths: length of cmunext blocks
                kernels: kernal size of cmunext blocks
            """
            super(CMUNeXt, self).__init__()
            # Encoder
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
            self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
            self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
            self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
            self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
            self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
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
            
            self.rwkv = BinaryOrientatedRWKV2D(n_embd=dims[-1],n_layer=8)
            self.rwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[-2],n_layer=8)


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

    return CMUNeXt()

def cmunext_chin_4_no_fusion_dec_se_rwkv45(reduction = 4):
    class fusion_conv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(fusion_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1)),
                # nn.GELU(),
                # nn.BatchNorm2d(ch_out * 4),
                # nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )
            self.se = SELayer(ch_out, reduction=reduction)
        def forward(self, x):
            x = self.conv(x)
            x = self.se(x)
            return x

    class CMUNeXt(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
            """
            Args:
                input_channel : input channel.
                num_classes: output channel.
                dims: length of channels
                depths: length of cmunext blocks
                kernels: kernal size of cmunext blocks
            """
            super(CMUNeXt, self).__init__()
            # Encoder
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
            self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
            self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
            self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
            self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
            self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
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
            
            self.rwkv = BinaryOrientatedRWKV2D(n_embd=dims[-1],n_layer=8)
            self.rwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[-2],n_layer=8)


        def forward(self, x):
            x1 = self.stem(x)
            x1 = self.encoder1(x1)
            x2 = self.Maxpool(x1)
            x2 = self.encoder2(x2)
            x3 = self.Maxpool(x2)
            x3 = self.encoder3(x3)
            x4 = self.Maxpool(x3)
            x4 = self.encoder4(x4)

            x4 = self.rwkv_4(x4)
            
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

    return CMUNeXt()

def cmunext_rwkv45():
    class CMUNeXt(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
            """
            Args:
                input_channel : input channel.
                num_classes: output channel.
                dims: length of channels
                depths: length of cmunext blocks
                kernels: kernal size of cmunext blocks
            """
            super(CMUNeXt, self).__init__()
            # Encoder
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
            self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
            self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
            self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
            self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
            self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
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
            
            self.rwkv = BinaryOrientatedRWKV2D(n_embd=dims[-1],n_layer=8)
            self.rwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[-2],n_layer=8)


        def forward(self, x):
            x1 = self.stem(x)
            x1 = self.encoder1(x1)
            x2 = self.Maxpool(x1)
            x2 = self.encoder2(x2)
            x3 = self.Maxpool(x2)
            x3 = self.encoder3(x3)
            x4 = self.Maxpool(x3)
            x4 = self.encoder4(x4)

            x4 = self.rwkv_4(x4)
            
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

    return CMUNeXt()

def cmunext_rwkv5():
    class CMUNeXt(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
            """
            Args:
                input_channel : input channel.
                num_classes: output channel.
                dims: length of channels
                depths: length of cmunext blocks
                kernels: kernal size of cmunext blocks
            """
            super(CMUNeXt, self).__init__()
            # Encoder
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
            self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
            self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
            self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
            self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
            self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
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
            
            self.rwkv = BinaryOrientatedRWKV2D(n_embd=dims[-1],n_layer=8)
            self.rwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[-2],n_layer=8)


        def forward(self, x):
            x1 = self.stem(x)
            x1 = self.encoder1(x1)
            x2 = self.Maxpool(x1)
            x2 = self.encoder2(x2)
            x3 = self.Maxpool(x2)
            x3 = self.encoder3(x3)
            x4 = self.Maxpool(x3)
            x4 = self.encoder4(x4)

            # x4 = self.rwkv_4(x4)
            
            x5 = self.Maxpool(x4)
            x5 = self.encoder5(x5)

            # x5 = self.rwkv(x5)
            
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

    return CMUNeXt()


def cmunext_rwkv45_allchannel_shift(channel = 'all'):
    
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
        if channel == 'all':
            # print("q_shift is called! ALL Channels") 
            output[:, :, :, shift_pixel:W] = input[:, :, :, 0:W-shift_pixel]
            output[:, :, :, 0:W-shift_pixel] = input[:, :, :, shift_pixel:W]
            output[:, :, shift_pixel:H, :] = input[:, :, 0:H-shift_pixel, :]
            output[:, :, 0:H-shift_pixel, :] = input[:, :, shift_pixel:H, :]
            # output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
        else:
            output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
            output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
            output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
            output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
            output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
        return output.flatten(2).transpose(1, 2)    
    
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
                    hidden_rate=4, init_mode='fancy', drop_path=0., key_norm=True, ffn_first=False, use_gc=False):
            super().__init__()
            self.use_gc = use_gc

            # # 定义 GSC 模块
            # if use_gc:
            #     self.gsc = GSC(n_embd)

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

            # # 使用 GSC
            # if self.use_gc:
            #     z = self.gsc(z)

            z_f = z.flatten(2).transpose(1, 2)
            z_r = z.transpose(2, 3).flatten(2).transpose(1, 2)

            rwkv_f = self.block_forward_rwkv(z_f)
            rwkv_r = self.block_backward_rwkv(z_r)

            fused = rwkv_f + rwkv_r
            if len(fused.shape) == 3:
                fused = fused.transpose(1, 2).view(B, C, H, W)
            return fused

    class CMUNeXt(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
            """
            Args:
                input_channel : input channel.
                num_classes: output channel.
                dims: length of channels
                depths: length of cmunext blocks
                kernels: kernal size of cmunext blocks
            """
            super(CMUNeXt, self).__init__()
            # Encoder
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
            self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
            self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
            self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
            self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
            self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
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
            
            self.rwkv = BinaryOrientatedRWKV2D(n_embd=dims[-1],n_layer=8)
            self.rwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[-2],n_layer=8)


        def forward(self, x):
            x1 = self.stem(x)
            x1 = self.encoder1(x1)
            x2 = self.Maxpool(x1)
            x2 = self.encoder2(x2)
            x3 = self.Maxpool(x2)
            x3 = self.encoder3(x3)
            x4 = self.Maxpool(x3)
            x4 = self.encoder4(x4)

            # x4 = self.rwkv_4(x4)
            
            x5 = self.Maxpool(x4)
            x5 = self.encoder5(x5)

            # x5 = self.rwkv(x5)
            
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
    return CMUNeXt()



def q_shift(x, shift_pixel=1, gamma=1/4, resolution=None, groups=8):
    """
    对 4D 张量 (BCHW) 进行分组八面移位操作。
    :param x: 输入张量，形状为 (B, C, H, W)
    :param groups: 分组数，默认为 8
    :return: 位移后的特征图，形状为 (B, C * 8, H, W)
    """
    
    B,L, C= x.shape
    assert C % groups == 0, "通道数必须能被分组数整除"
    channels_per_group = C // groups
    
    if resolution is None:
        sqrt_L = int(L ** 0.5)
        if sqrt_L * sqrt_L == L:
            resolution = (sqrt_L, sqrt_L)
        else:
            raise ValueError("Cannot infer a valid resolution for the given input shape.")
    # 将输入张量按通道分组
    
    x_grouped = x.view(B, groups, channels_per_group, resolution[0], resolution[1])

    # 定义 8 种位移方向
    shifts = [
        (0, 0),    # 不位移
        (0, 1),    # 宽度方向右移
        (0, -1),   # 宽度方向左移
        (1, 0),    # 高度方向下移
        (-1, 0),   # 高度方向上移
        (1, 1),    # 右下方向位移
        (1, -1),   # 左下方向位移
        (-1, 1),   # 右上方向位移
        (-1, -1),  # 左上方向位移
    ]

    # 对每一组进行八面移位
    shifted_features = []
    for h_shift, w_shift in shifts:
        
        shifted_x = torch.roll(x_grouped, shifts=(h_shift, w_shift), dims=(-2, -1))
        shifted_features.append(shifted_x)

    # 将位移结果按通道维度拼接
    shifted_features = torch.cat(shifted_features, dim=2)  # (B, groups, C * 8, H, W)
    # return 4d
    # shifted_features = shifted_features.view(B, -1, resolution[0], resolution[1])  # (B, C * 8, H, W)

    # return 3d
    shifted_features = shifted_features.view(B,  L, -1)
    return shifted_features

def cmunext_rwkv45_octo_shift():
    
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

                # # 创建一个随机的 4D 张量 (N, C, H, W)
                # N, C, H, W = 2, 16, 8, 8  # 批量大小为 2，通道数为 16，高度和宽度为 8
                # input_tensor = torch.randn(N, C, H, W)

                # print("输入张量形状:", input_tensor.shape)
                # print("输入张量内容 (第一个样本的第一个通道):")
                # print(input_tensor[0, 0])  # 打印第一个样本的第一个通道

                # # 调用 q_shift 函数
                # output_tensor = self.shift_func(input_tensor)
                # # print("U in")
                # print(output_tensor.shape)
                xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, resolution)
                # print(f"x shape: {x.shape}")
                # print(f"xx shape: {xx.shape}")                
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
                    init_mode='fancy', shift_mode = 'q_shift', key_norm=False,ffn_first=False):
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
                shift_mode=shift_mode,
                key_norm=key_norm
            )
            
            # 频域混合模块
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode=shift_mode,
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
                    hidden_rate=4, init_mode='fancy', drop_path=0., key_norm=True, ffn_first=False, use_gc=False):
            super().__init__()
            self.use_gc = use_gc

            # # 定义 GSC 模块
            # if use_gc:
            #     self.gsc = GSC(n_embd)

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

            # # 使用 GSC
            # if self.use_gc:
            #     z = self.gsc(z)

            z_f = z.flatten(2).transpose(1, 2)
            z_r = z.transpose(2, 3).flatten(2).transpose(1, 2)

            rwkv_f = self.block_forward_rwkv(z_f)
            rwkv_r = self.block_backward_rwkv(z_r)

            fused = rwkv_f + rwkv_r
            if len(fused.shape) == 3:
                fused = fused.transpose(1, 2).view(B, C, H, W)
            return fused

    class CMUNeXt(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
            """
            Args:
                input_channel : input channel.
                num_classes: output channel.
                dims: length of channels
                depths: length of cmunext blocks
                kernels: kernal size of cmunext blocks
            """
            super(CMUNeXt, self).__init__()
            # Encoder
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
            self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
            self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
            self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
            self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
            self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
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
            
            self.rwkv = BinaryOrientatedRWKV2D(n_embd=dims[-1],n_layer=8)
            self.rwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[-2],n_layer=8)


        def forward(self, x):
            x1 = self.stem(x)
            x1 = self.encoder1(x1)
            x2 = self.Maxpool(x1)
            x2 = self.encoder2(x2)
            x3 = self.Maxpool(x2)
            x3 = self.encoder3(x3)
            x4 = self.Maxpool(x3)
            x4 = self.encoder4(x4)

            # x4 = self.rwkv_4(x4)
            
            x5 = self.Maxpool(x4)
            x5 = self.encoder5(x5)
            
            # print("Before rwkv x5.shape", x5.shape)
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
    return CMUNeXt()

def q_shift(x, shift_pixel=1, gamma=1/4, resolution=None, groups=8):
    """
    对 3D 张量进行分组八面移位操作。
    :param x: 输入张量，形状为 (B, L, C)
    :param groups: 分组数，默认为 8
    :return: 位移后的特征图，形状为 (B, L, C * 8)
    """
    B, L, C = x.shape
    assert C % groups == 0, "通道数必须能被分组数整除"
    channels_per_group = C // groups

    if resolution is None:
        sqrt_L = int(L ** 0.5)
        if sqrt_L * sqrt_L == L:
            resolution = (sqrt_L, sqrt_L)
        else:
            raise ValueError("Cannot infer a valid resolution for the given input shape.")

    # 将输入张量按通道分组并转换为 4D (B, groups, C//groups, H, W)
    x_grouped = x.view(B, groups, channels_per_group, resolution[0], resolution[1])

    # 定义 8 种位移方向
    shifts = [
        (0, 1),    # 宽度方向右移
        (0, -1),   # 宽度方向左移
        (1, 0),    # 高度方向下移
        (-1, 0),   # 高度方向上移
        (1, 1),    # 右下方向位移
        (1, -1),   # 左下方向位移
        (-1, 1),   # 右上方向位移
        (-1, -1),  # 左上方向位移
    ]

    # 对每一组进行八面移位，并将结果 reshape 为 (B, L, C)
    shifted_features = []
    for h_shift, w_shift in shifts:
        shifted_x = torch.roll(x_grouped, shifts=(h_shift, w_shift), dims=(-2, -1))  # (B, groups, C//groups, H, W)
        shifted_x = shifted_x.view(B, L, C)  # (B, L, C)
        shifted_features.append(shifted_x)

    # 将位移结果按通道维度拼接
    # shifted_features = torch.cat(shifted_features, dim=-1)  # (B, L, C * 8)
    return shifted_features

def cmunext_rwkv45_octo_shift():

    class SpatialMix(nn.Module):
        def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                    channel_gamma=1/4, shift_pixel=1, key_norm=True):
            super().__init__()
            self.layer_id = layer_id
            self.n_layer = n_layer
            self.n_embd = n_embd
            self.device = None

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

            # 定义线性层
            self.key = nn.Linear(n_embd, n_embd, bias=False)
            self.value = nn.Linear(n_embd, n_embd, bias=False)
            self.receptance = nn.Linear(n_embd, n_embd, bias=False)

            # 可选的 LayerNorm
            if key_norm:
                self.key_norm = nn.LayerNorm(n_embd)
            else:
                self.key_norm = None

            # 输出线性层
            self.output = nn.Linear(n_embd, n_embd, bias=False)

        def forward(self, x):
            B, T, C = x.size()
            self.device = x.device

            # 计算 sr, k, v
            sr, k, v = self.jit_func(x)
            rwkv = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
            if self.key_norm is not None:
                rwkv = self.key_norm(rwkv)
            rwkv = sr * rwkv
            rwkv = self.output(rwkv)
            return rwkv

        def jit_func(self, x):
            B, T, C = x.size()
            xx = x  # 这里可以根据需要添加 def q_shift 逻辑
            # print("spatialmix xshape",x.shape)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)

            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)

            return sr, k, v


        
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
            # xx = torch.cat(x, dim=0)
            # xx =  torch.stack(x, dim=-1)
            # xx = xx.view(xx.size(0), xx.size(1), 8, -1)  # (B, T, 8, 256)
            # x = xx.mean(dim=2)  # (B, T, 256)  

            if self.shift_pixel > 0: # def q_shift

                # xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, resolution)  # (B, T, 2048)
                # xx = x.view(x.size(0), x.size(1), 8, -1)  # (B, T, 8, 256)
                # xx = xx.mean(dim=2)  # (B, T, 256)

                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            else:
                xk = x
                xr = x

            # 计算 key、value 和 receptance
            # print(xk.shape)
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
                    init_mode='fancy', shift_mode='q_shift', key_norm=False, ffn_first=False):
            super().__init__()
            self.layer_id = layer_id
            self.depth = depth
            self.ffn_first = ffn_first

            # LayerNorm 层
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

            # 空间混合模块
            self.att = nn.ModuleList([
                SpatialMix(
                    n_embd=n_embd,
                    n_layer=n_layer,
                    layer_id=i,
                    shift_mode=shift_mode,
                    key_norm=key_norm
                ) for i in range(8)  # 8 个 SpatialMix 模块
            ])

            # 频域混合模块
            self.ffn = SpectralMixer(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                shift_mode=shift_mode,
                shift_pixel=-1,
                key_norm=key_norm
            )

            # 可学习的缩放参数
            self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
            self.gamma8 = nn.Parameter(torch.ones((n_embd * 8)), requires_grad=True)

        def forward(self, x):
            if len(x.shape) == 4:
                b, c, h, w = x.shape
                x = rearrange(x, 'b c h w -> b (h w) c')
            b, n, c = x.shape
            h = w = int(n ** 0.5)
            resolution = (h, w)
            # 空间混合
            # if self.ffn_first:
            # print(x.shape)
            _, x = self.apply_spatial_mix(self.ln1(x))
            # print(type(x_mid))
            # for i in range(8):
                # x = x + self.ffn(self.ln2(spa_outs[i]))
            # x = x + self.gamma1 * self.apply_spatial_mix(self.ln1(x))
            # else: 
            #     x = x + self.gamma1 * self.apply_spatial_mix(self.ln1(x))
            #     x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)

            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x

        def apply_spatial_mix(self, x):
            # 对 q_shift 的输出进行处理
            xx = q_shift(x)  # (B, 256*8, 256) 是一个 [list]
            # print(f"q_shift output shape: {xx.shape}") 
            # xx = xx.view(x.size(0), 8, 256, 256)  # (B, 8, 256, 256)

            # 对每个子张量分别通过 SpatialMix
            outputs = []
            ys = []
            # import pdb;pdb.set_trace()
            
            for i in range(8):
                output = self.att[i](self.ln1(xx[i]))  # (B, 256, 256)
                outputs.append(output)
                
                y = xx[i] + self.gamma1 * output
                y = y + self.gamma2 * self.ffn(self.ln2(y))
                # y = self.ln2(y)
                # ys.append(y)

            # # 合并结果
            # outputs = torch.stack(outputs, dim=1)  # (B, 8, 256, 256)
            # outputs = outputs.view(x.size(0), 256 , 256* 8)  # (B, 256, 256*8)
            # return outputs
            # y = 0 + self.gamma2 * self.ffn(ys)
            ##  self.ffn 是 chin C*8 所以 append 进去 for 循环出来之后再开始。            
            
            return outputs, y

    class BinaryOrientatedRWKV2D(nn.Module):
        def __init__(self, n_embd, n_layer, shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1, 
                    hidden_rate=4, init_mode='fancy', drop_path=0., key_norm=True, ffn_first=False, use_gc=False):
            super().__init__()
            self.use_gc = use_gc

            # # 定义 GSC 模块
            # if use_gc:
            #     self.gsc = GSC(n_embd)

            self.block_forward_rwkv = RWKVBlock(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=0,
                hidden_rate=8,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )
            self.block_backward_rwkv = RWKVBlock(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=1,
                hidden_rate=8,
                init_mode=init_mode,
                key_norm=key_norm,
                ffn_first=ffn_first
            )

        def forward(self, z):
            B, C, H, W = z.shape

            # # 使用 GSC
            # if self.use_gc:
            #     z = self.gsc(z)

            z_f = z.flatten(2).transpose(1, 2)
            z_r = z.transpose(2, 3).flatten(2).transpose(1, 2)

            rwkv_f = self.block_forward_rwkv(z_f)
            rwkv_r = self.block_backward_rwkv(z_r)

            fused = rwkv_f + rwkv_r
            if len(fused.shape) == 3:
                fused = fused.transpose(1, 2).view(B, C, H, W)
            return fused

    class CMUNeXt(nn.Module):
        def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160,256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
            """
            Args:
                input_channel : input channel.
                num_classes: output channel.
                dims: length of channels
                depths: length of cmunext blocks
                kernels: kernal size of cmunext blocks
            """
            super(CMUNeXt, self).__init__()
            # Encoder
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
            self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
            self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
            self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
            self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
            self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
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
            
            self.rwkv = BinaryOrientatedRWKV2D(n_embd=dims[-1],n_layer=8)
            self.rwkv_4 = BinaryOrientatedRWKV2D(n_embd=dims[-2],n_layer=8)


        def forward(self, x):
            x1 = self.stem(x)
            x1 = self.encoder1(x1)
            x2 = self.Maxpool(x1)
            x2 = self.encoder2(x2)
            x3 = self.Maxpool(x2)
            x3 = self.encoder3(x3)
            x4 = self.Maxpool(x3)
            x4 = self.encoder4(x4)

            x4 = self.rwkv_4(x4)
            
            x5 = self.Maxpool(x4)
            x5 = self.encoder5(x5)
            
            # print("Before rwkv x5.shape", x5.shape)
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
    return CMUNeXt()

if __name__ == "__main__":
    from thop import profile,clever_format
    x = torch.randn(1, 3, 256, 256).cuda()
    # model = LoRD(dims=[32, 64, 128, 256,768]).cuda()
    # model = CMUNeXt(dims=[24, 48, 96, 192,384]).cuda()
    # model = cmunext().cuda()
    # model = cmunext_no_chin_4().cuda()
    # model = cmunext_no_chout_4_fusion().cuda()
    # model = cmunext_no_chout_4_fusion().cuda()
    # model = cmunext_rwkv45_allchannel_shift().cuda()
    model = cmunext_rwkv45_octo_shift().cuda()


    print(model(x).shape)
    flops, params = profile(model,inputs=(x,))
    flops, params  = clever_format((flops,params),"%.2f")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")