import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


# 假设的 ViT 参数
image_size = 224
patch_size = 16
channels = 3
dim = 768

# 计算 patch_dim
patch_height, patch_width = patch_size, patch_size
patch_dim = channels * patch_height * patch_width

# 创建 to_patch_embedding 模块
to_patch_embedding = nn.Sequential(
    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
    nn.LayerNorm(patch_dim),
    nn.Linear(patch_dim, dim),
    nn.LayerNorm(dim),
)

# 创建一个随机图像
batch_size = 2
img = torch.randn(batch_size, channels, image_size, image_size)

# 应用 to_patch_embedding
print("Input image shape:", img.shape)
x = to_patch_embedding[0](img)
print("Shape after Rearrange:", x.shape)
x = to_patch_embedding[1](x)
print("Shape after LayerNorm(patch_dim):", x.shape)
x = to_patch_embedding[2](x)
print("Shape after Linear(patch_dim, dim):", x.shape)
x = to_patch_embedding[3](x)
print("Shape after LayerNorm(dim):", x.shape)