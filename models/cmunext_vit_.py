import torch
import torch.nn as nn
import argparse

from timm.models.layers import DropPath


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
        if sr_ratio > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
            self.LocalProp = nn.ConvTranspose2d(dim, dim, sr_ratio, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.LocalProp = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.sr > 1:
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
            x = x.permute(0, 2, 1).reshape(B, C, H // self.sr, W // self.sr)
            x = self.LocalProp(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., sr_ratio=1.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GlobalSparseAttn(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class VitBottleNeck(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout=0.001, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, padding=1),
            nn.Flatten(2)
        )
        self.SelfAttn = SelfAttn(embed_dim, num_heads, mlp_ratio)

    def forward(self, x):
        B, N, H, W = x.shape
        x = self.SelfAttn(x)
        return x


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


class CMUNeXt_vit_1_3_128_256_768(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 256, 768], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7], args_vit=None):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(2, 2)
        self.stem = conv_block(input_channel, dims[0])
        self.encoder1 = CMUNeXtBlock(dims[0], dims[0], depths[0], kernels[0])
        self.encoder2 = CMUNeXtBlock(dims[0], dims[1], depths[1], kernels[1])
        self.encoder3 = CMUNeXtBlock(dims[1], dims[2], depths[2], kernels[2])
        self.encoder4 = CMUNeXtBlock(dims[2], dims[3], depths[3], kernels[3])
        self.encoder5 = CMUNeXtBlock(dims[3], dims[4], depths[4], kernels[4])

        PATCH_SIZE = args_vit.patch_size if args_vit else 1
        NUM_PATCHES = (16 // PATCH_SIZE) ** 2
        IN_CHANNELS = dims[4]
        EMBED_DIM = dims[4]

        self.vit = VitBottleNeck(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES)

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

        x5 = self.vit(x5)

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


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=1)
    return parser


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = get_arg_parser()
    args_vit = parser.parse_args()
    model = CMUNeXt_vit_1_3_128_256_768(input_channel=3, args_vit=args_vit, num_classes=1)
    x = torch.randn(2, 3, 256, 256)
    print(count_params(model))
    output = model(x)
    print(output.shape)