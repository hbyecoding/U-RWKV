from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import 





class GlobalSparseAttn(nn.Module):
    
    def __init__(self, dim, num_heads = 8, qkv_bias=False, qk_scale = None, attn_drop = 0., proj_drop = 0., sr_ratio=1):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or  head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr= sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
            kernel_size = sr_ratio
            self.LocalProp= nn.ConvTranspose2d(dim, dim, kernel_size, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        if self.sr > 1.:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(H/self.sr), int(W/self.sr))
            x = self.LocalProp(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LocalAgg(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv-bias = False, qk_scale = None, drop=0)
        self.norm1 = nn.BatchNorm2d(dim)
        ## self conv1 conv2 
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        
        self.attn = nn.Conv2d(dim,dim, 5, padding = 2, groups = dim)
    def forward(self,x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio =4., qkv_bias = False, qk_scale = None, 
                 drop = 0., attn_drop = 0., drop_path = 0., act_layer= nn.GELU, 
                 norm_layer = nn.LayerNorm, sr_ratio=1.):
        
        super().__init__()
        self.attn = GlobalSparseAttn(
            dim, 
            num_heads = num_heads,
            qkv_bias = qkv_bias,
            qk_scale = qk_scale,
            attn_drop= attn_drop,
            proj_drop = drop,
            sr_ratio= sr_ratio
        )


import torch
import torch.nn.functional as F
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query, key, value):
        B, N, C = query.shape
        query = self.query(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        return self.out(attn_output)        

def q_shift(input, shift_pixel = 1, gamma = 1/4, patch_resolution = None)        :
    
    assert gamma <=1/4
    B,N,C = input.shape
    
    if patch_resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            patch_resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError("Cannot infer a valid patch_resolution for the given input shape.")

    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    
    output = torch.zeros_like(input)
    
    # 简单线形层
    
    # attn_layer = torch.nn.Linear(9,9)
    
    # for i in range(shift_pixel, H - shift_pixel):
    #     for j in range(shift_pixel, W - shift_pixel):
    #         # 提取周边8个像素

    # 定义多头注意力机制
    mha = MultiHeadAttention(embed_dim=C, num_heads=4)
    
    # 对每个像素点的周边八个像素进行加权平均
    for i in range(shift_pixel, H - shift_pixel):
        for j in range(shift_pixel, W - shift_pixel):
            # 提取周边八个像素
            patch = input[:, :, i-shift_pixel:i+shift_pixel+1, j-shift_pixel:j+shift_pixel+1]
            patch = patch.flatten(2, 3)  # 将3x3的patch展平为(B, C, 9)
            patch = patch.transpose(1, 2)  # (B, 9, C)
            
            # 计算多头注意力分数
            attn_output = mha(patch, patch, patch)  # (B, 9, C)
            
            # 将加权平均结果赋值给输出张量
            output[:, :, i, j] = attn_output[:, 4, :]  # 取中心像素的加权结果
    
    # 处理边界情况
    output[:, :, :shift_pixel, :] = input[:, :, :shift_pixel, :]
    output[:, :, H-shift_pixel:, :] = input[:, :, H-shift_pixel:, :]
    output[:, :, :, :shift_pixel] = input[:, :, :, :shift_pixel]
    output[:, :, :, W-shift_pixel:] = input[:, :, :, W-shift_pixel:]
    
    # 返回结果
    return output.flatten(2).transpose(1, 2)


# center_pixel = patch[:, :, ]
            
        