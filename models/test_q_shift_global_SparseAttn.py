import torch
import torch.nn.functional as F
from torch import nn

# GlobalSparseAttn 模块
class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
            kernel_size = sr_ratio
            self.LocalProp = nn.ConvTranspose2d(dim, dim, kernel_size, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.LocalProp = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        if self.sr > 1:
            x = x.transpose(1, 2).reshape(B, C, int(N ** 0.5), int(N ** 0.5))
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(N ** 0.5 / self.sr), int(N ** 0.5 / self.sr))
            x = self.LocalProp(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn  # 返回注意力分数


# 修改后的 q_shift 函数
def q_shift_with_global_sparse_attn(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
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

    # 创建输出张量
    output = torch.zeros_like(input)

    # 定义 GlobalSparseAttn 模块
    global_sparse_attn = GlobalSparseAttn(dim=C, num_heads=4, sr_ratio=1)  # 使用 sr_ratio=1 不进行下采样

    # 对每个像素点的周边八个像素进行加权平均
    for i in range(shift_pixel, H - shift_pixel):
        for j in range(shift_pixel, W - shift_pixel):
            # 提取周边八个像素
            patch = input[:, :, i-shift_pixel:i+shift_pixel+1, j-shift_pixel:j+shift_pixel+1]
            patch = patch.flatten(2, 3)  # 将3x3的patch展平为(B, C, 9)
            patch = patch.transpose(1, 2)  # (B, 9, C)

            # 计算注意力分数
            attn_output, attn_scores = global_sparse_attn(patch)  # (B, 9, C), (B, num_heads, 9, 9)

            # 对注意力分数进行归一化
            attn_scores = attn_scores.mean(dim=1)  # 对多头注意力分数取平均 (B, 9, 9)
            attn_scores = attn_scores[:, 4, :]  # 取中心像素对周边像素的注意力分数 (B, 9)
            attn_scores = F.softmax(attn_scores, dim=-1)  # 归一化

            # 加权平均
            weighted_patch = torch.einsum('bij,bj->bij', patch, attn_scores)  # (B, 9, C)

            # 将加权平均结果赋值给输出张量
            output[:, :, i, j] = weighted_patch[:, 4, :]  # 取中心像素的加权结果

    # 处理边界情况
    output[:, :, :shift_pixel, :] = input[:, :, :shift_pixel, :]
    output[:, :, H-shift_pixel:, :] = input[:, :, H-shift_pixel:, :]
    output[:, :, :, :shift_pixel] = input[:, :, :, :shift_pixel]
    output[:, :, :, W-shift_pixel:] = input[:, :, :, W-shift_pixel:]

    # 返回结果
    return output.flatten(2).transpose(1, 2)
# 基于余弦相似度的 q_shift
def q_shift_cosine(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
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
    
    # 创建输出张量
    output = torch.zeros_like(input)
    
    # 对每个像素点的周边八个像素进行加权平均
    for i in range(shift_pixel, H - shift_pixel):
        for j in range(shift_pixel, W - shift_pixel):
            # 提取周边八个像素
            patch = input[:, :, i-shift_pixel:i+shift_pixel+1, j-shift_pixel:j+shift_pixel+1]
            patch = patch.flatten(2, 3)  # 将3x3的patch展平为(B, C, 9)
            
            # 计算余弦相似度作为注意力分数
            center_pixel = patch[:, :, 4].unsqueeze(2)  # 中心像素 (B, C, 1)
            other_pixels = patch[:, :, [0, 1, 2, 3, 5, 6, 7, 8]]  # 周边8个像素 (B, C, 8)
            
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(center_pixel, other_pixels, dim=1)  # (B, 8)
            cos_sim = torch.cat([torch.zeros(B, 1, device=input.device), cos_sim], dim=1)  # 添加中心像素的分数 (B, 9)
            
            # 归一化注意力分数
            attn_scores = F.softmax(cos_sim, dim=-1)  # (B, 9)
            
            # 加权平均
            weighted_patch = torch.einsum('bij,bj->bij', patch, attn_scores)  # (B, C, 9)
            
            # 将加权平均结果赋值给输出张量
            output[:, :, i, j] = weighted_patch[:, :, 4]  # 取中心像素的加权结果
    
    # 处理边界情况
    output[:, :, :shift_pixel, :] = input[:, :, :shift_pixel, :]
    output[:, :, H-shift_pixel:, :] = input[:, :, H-shift_pixel:, :]
    output[:, :, :, :shift_pixel] = input[:, :, :, :shift_pixel]
    output[:, :, :, W-shift_pixel:] = input[:, :, :, W-shift_pixel:]
    
    # 返回结果
    return output.flatten(2).transpose(1, 2)

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

def q_shift_mha(input, shift_pixel = 1, gamma = 1/4, patch_resolution = None)        :
    
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

## method 1 q_shift in fact is  omnishift
def q_shift_1_5(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
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
    
    # 创建输出张量
    output = torch.zeros_like(input)
    
    # 定义周边八个像素的权重
    weights = torch.tensor([
        [1/20, 1/5, 1/20],
        [1/5, 0, 1/5],
        [1/20, 1/5, 1/20]
    ], dtype=input.dtype, device=input.device)
    
    # 对每个像素点的周边八个像素进行加权平均
    for i in range(shift_pixel, H - shift_pixel):
        for j in range(shift_pixel, W - shift_pixel):
            # 提取周边八个像素
            patch = input[:, :, i-shift_pixel:i+shift_pixel+1, j-shift_pixel:j+shift_pixel+1]
            # 加权平均
            weighted_patch = torch.einsum('bchw,hw->bc', patch, weights)
            # 将加权平均结果赋值给输出张量
            output[:, :, i, j] = weighted_patch
    
    # 处理边界情况
    output[:, :, :shift_pixel, :] = input[:, :, :shift_pixel, :]
    output[:, :, H-shift_pixel:, :] = input[:, :, H-shift_pixel:, :]
    output[:, :, :, :shift_pixel] = input[:, :, :, :shift_pixel]
    output[:, :, :, W-shift_pixel:] = input[:, :, :, W-shift_pixel:]
    
    # 返回结果
    return output.flatten(2).transpose(1, 2)

## method 1 q_shift in fact is  omnishift
def q_shift_1_8(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
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
    
    # 创建输出张量
    output = torch.zeros_like(input)
    
    # 定义周边八个像素的权重
    weights = torch.tensor([
        [1/8, 1/8, 1/8],
        [1/8, 0, 1/8],
        [1/8, 1/8, 1/8]
    ], dtype=input.dtype, device=input.device)
    
    # 对每个像素点的周边八个像素进行加权平均
    for i in range(shift_pixel, H - shift_pixel):
        for j in range(shift_pixel, W - shift_pixel):
            # 提取周边八个像素
            patch = input[:, :, i-shift_pixel:i+shift_pixel+1, j-shift_pixel:j+shift_pixel+1]
            # 加权平均
            weighted_patch = torch.einsum('bchw,hw->bc', patch, weights)
            # 将加权平均结果赋值给输出张量
            output[:, :, i, j] = weighted_patch
    
    # 处理边界情况
    output[:, :, :shift_pixel, :] = input[:, :, :shift_pixel, :]
    output[:, :, H-shift_pixel:, :] = input[:, :, H-shift_pixel:, :]
    output[:, :, :, :shift_pixel] = input[:, :, :, :shift_pixel]
    output[:, :, :, W-shift_pixel:] = input[:, :, :, W-shift_pixel:]
    
    # 返回结果
    return output.flatten(2).transpose(1, 2)

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    B, C, H, W = 1, 3, 16, 16  # 批次大小为1，通道数为3，分辨率为16x16
    input = torch.rand(B, H * W, C)  # 随机生成输入数据
    
    # 调用基于余弦相似度的 q_shift
    output_cosine = q_shift_cosine(input, patch_resolution=(H, W))
    
    # 调用基于多头注意力机制的 q_shift
    output_mha = q_shift_mha(input, patch_resolution=(H, W))
    
    output_1_8 = q_shift_1_8(input, patch_resolution=(H, W))
    output_1_5 = q_shift_1_5(input, patch_resolution=(H, W))
    
    # 可视化结果并保存为图像
    def save_images(input, output, prefix, save_path='result'):
        
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        input = input.reshape(B, C, H, W).permute(0, 2, 3, 1).squeeze(0).numpy()
        output = output.reshape(B, C, H, W).permute(0, 2, 3, 1).squeeze(0).numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Input")
        plt.imshow(input)
        plt.subplot(1, 2, 2)
        plt.title("Output")
        plt.imshow(output)

        # 保存图像
        plt.savefig(f"{save_path}/{prefix}_result.png")
        plt.close()

    # # 创建保存目录
        # import os
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

    # # 保存图像
    # save_images(input, output, save_path, "global_sparse_attn")
    
    save_images(input, output_cosine, "Cosine Similarity")
    save_images(input, output_1_8, "Cosine Similarity")
    # visualize(input, output_mha, "Multi-Head Attention")