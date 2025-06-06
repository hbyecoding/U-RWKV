import os, sys
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch_dwconv import DepthwiseConv2d
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcls.models.utils import resize_pos_embed
from mmcls.models.necks.gap import GlobalAveragePooling
from mmcls.models.backbones.base_backbone import BaseBackbone
T_MAX = 640 #128*128 2048 均不可以 # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load


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
        elif len(x.shape) == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
        # B = x.shape[0]    
        x = x + self.drop_path(self.spatial_mix(self.ln1(x), patch_resolution))

        # Channel Mixing (FFN-like)
        x = x + self.drop_path(self.channel_mix(self.ln2(x), patch_resolution))
        if len(x.shape) == 3:
            out = x.transpose(1, 2).reshape(B, C, H, W)
        elif len(x.shape) == 4:
            out = out
        return out

# 自动填充函数
def autopad(k, p=None, d=1):
    """
    k: kernel
    p: padding
    d: dilation
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

# 标准卷积模块
class Conv(nn.Module):
    default_act = nn.GELU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class DSWConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, groups, ks=3, padding=0, act='gelu', bias=False):
        super().__init__()
        self.dwconv2d = Residual(nn.Sequential(
            DepthwiseConv2d(ch_in, ch_in, kernel_size=ks, padding=padding, groups=groups, bias=bias),
            nn.GELU() if act.lower() == 'gelu' else nn.ReLU() if act.lower() == 'relu' else nn.LeakyReLU(0.1) if act.lower() == 'leakyrelu' else nn.SiLU() if act.lower() == 'silu' else nn.Identity(),
            nn.BatchNorm2d(ch_in)
        ))
        self.conv1x1_expand = nn.Sequential(
            nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1), bias=True),
            nn.GELU() if act.lower() == 'gelu' else nn.ReLU() if act.lower() == 'relu' else nn.LeakyReLU(0.1) if act.lower() == 'leakyrelu' else nn.SiLU() if act.lower() == 'silu' else nn.Identity(),
            nn.BatchNorm2d(ch_in * 4)
        )
        self.conv1x1_compress = nn.Sequential(
            nn.Conv2d(ch_in * 4, ch_out, kernel_size=(1, 1), bias=True),
            nn.GELU() if act.lower() == 'gelu' else nn.ReLU() if act.lower() == 'relu' else nn.LeakyReLU(0.1) if act.lower() == 'leakyrelu' else nn.SiLU() if act.lower() == 'silu' else nn.Identity(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x = self.dwconv2d(x)
        x = self.conv1x1_expand(x)
        x = self.conv1x1_compress(x)
        return x

# class LoANBlock(nn.Module):
#     def __init__(self, ch_in, ch_out, N=8, depth=8, shortcut=True, k=3, sep_num=2, act='gelu'):
#         super().__init__()
#         self.N = N
#         self.c = int(ch_out / sep_num / self.N)
#         self.add = shortcut and ch_in == ch_out
#         self.pwconv1 = Conv(ch_in, ch_out // self.N, 1, 1, act=act)
#         self.pwconv2 = Conv(ch_out // sep_num, ch_out, 1, 1, act=act)
#         self.m = nn.ModuleList([DSWConv2d(self.c, self.c, groups=self.c, ks=k, padding=k // 2, act=act) for _ in range(self.N - 1)])

#     def forward(self, x):
#         x_residual = x
#         x = self.pwconv1(x)
#         x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
#         x.extend(m(x[-1]) for m in self.m)
#         x[0] = x[0] + x[1]
#         x.pop(1)
#         y = torch.cat(x, dim=1)
#         y = self.pwconv2(y)
#         return x_residual + y if self.add else y


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

class FusionConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FusionConv, self).__init__()
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
    def __init__(self, input_channel=3, num_classes=1, dims=[24, 48, 96, 192, 384], depths=[1,1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = ConvBlock(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = LoRDBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = LoRDBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = LoRDBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = LoRDBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = LoRDBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        self.Brwkv = BinaryOrientatedRWKV2D(n_embd=dims[-1], 
                                            n_layer=12, 
                                            shift_mode='q_shift',
                                            channel_gamma=1/4,
                                            shift_pixel=1,
                                            hidden_rate=4,
                                            init_mode="fancy",
                                            drop_path=0,
                                            k_norm=True)
        self.Up5 = UpConv(ch_in=dims[4], ch_out=dims[3])
        self.UpConv5 = FusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = UpConv(ch_in=dims[3], ch_out=dims[2])
        self.UpConv4 = FusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = UpConv(ch_in=dims[2], ch_out=dims[1])
        self.UpConv3 = FusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = UpConv(ch_in=dims[1], ch_out=dims[0])
        self.UpConv2 = FusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
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

        x5 = self.Brwkv(x5)
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)
        d1 = self.Conv_1x1(d2)
        
        return d1

class BinaryOrientatedRWKV2D(nn.Module):
    def __init__(self, n_embd, n_layer, shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy', drop_path=0., k_norm=True):
        super().__init__()
        self.rwkv_forward = VRWKV_Bottleneck(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=0,
            shift_mode=shift_mode,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel,
            hidden_rate=hidden_rate,
            init_mode=init_mode,
            drop_path=drop_path,
            k_norm=k_norm
        )
        self.rwkv_reverse = VRWKV_Bottleneck(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=1,
            shift_mode=shift_mode,
            channel_gamma=channel_gamma,
            shift_pixel=shift_pixel,
            hidden_rate=hidden_rate,
            init_mode=init_mode,
            drop_path=drop_path,
            k_norm=k_norm
        )

    def forward(self, z):
        B, C, H, W = z.shape
        z_f = z.flatten(2).transpose(1, 2)
        z_r = z.transpose(2, 3).flatten(2).transpose(1, 2)

        rwkv_f = self.rwkv_forward(z_f)
        rwkv_r = self.rwkv_reverse(z_r)

        fused = rwkv_f + rwkv_r
        if len(fused.shape) == 3:
            fused = fused.transpose(1, 2).view(B, C, H, W)
        return fused

# class LoRD_4plusDeep(nn.Module):
#     def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 256, 768], depths=[3, 3, 3, 3, 1], kernels=[3, 3, 7, 7, 7]):
#         super().__init__()
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.stem = ConvBlock(ch_in=input_channel, ch_out=dims[0])
#         self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
#         self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
#         self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
#         self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
#         self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

#         self.Up5 = UpConv(ch_in=dims[4], ch_out=dims[3])
#         self.UpConv5 = FusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
#         self.Up4 = UpConv(ch_in=dims[3], ch_out=dims[2])
#         self.UpConv4 = FusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
#         self.Up3 = UpConv(ch_in=dims[2], ch_out=dims[1])
#         self.UpConv3 = FusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
#         self.Up2 = UpConv(ch_in=dims[1], ch_out=dims[0])
#         self.UpConv2 = FusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
#         self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         x1 = self.stem(x)
#         x1 = self.encoder1(x1)
#         x2 = self.Maxpool(x1)
#         x2 = self.encoder2(x2)
#         x3 = self.Maxpool(x2)
#         x3 = self.encoder3(x3)
#         x4 = self.Maxpool(x3)
#         x4 = self.encoder4(x4)
#         x5 = self.Maxpool(x4)
#         x5 = self.encoder5(x5)

#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.UpConv5(d5)

#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.UpConv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.UpConv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.UpConv2(d2)
#         d1 = self.Conv_1x1(d2)

#         return d1

if __name__ == "__main__":
    from thop import profile,clever_format
    x = torch.randn(1, 3, 256, 256).cuda()
    # model = LoRD(dims=[32, 64, 128, 256,768]).cuda()
    model = LoRD(dims=[24, 48, 96, 192,384]).cuda()

    flops, params = profile(model,inputs=(x,))
    flops, params  = clever_format((flops,params),"%.2f")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
