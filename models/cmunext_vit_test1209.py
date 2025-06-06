import torch
import torch.nn as nn
import math
import argparse

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
 

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# bk 
# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
#         super(PatchEmbedding, self).__init__()
#         if patch_size <=4:
#             self.patcher = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=3, stride=1,padding=1),
#                 nn.Flatten(2)
#             )
#         else:
#             self.patcher = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
#             nn.Flatten(2)
#         )

#         self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
#         self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, x):
#         y = x
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)

#         x = self.patcher(x).permute(0, 2, 1)
#         x = torch.cat([cls_token, x], dim=1)
#         x = x + self.position_embedding
#         x = self.dropout(x)
#         return x

class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,  sr_ratio=1):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj =  nn.Linear(dim, dim)
        
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr = sr_ratio
        if self.sr > 1 :
            self.sampler = nn.AvgPool2d(1, sr_ratio)
            kernel_size = sr_ratio
            self.LocalProp = nn.ConvTranspose2d(dim,
                                                dim, 
                                                kernel_size,
                                                stride=sr_ratio,
                                                groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()
    
    def forward(self, x, H:int, W:int):
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

class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio) #1
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # global layer_scale
        # self.ls = layer_scale

    def forward(self, x):
        # x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x
            

class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
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
        x = self.drop(x)
        return x        
    

class VitBottleNeck(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches,drop_path = 0.,  dropout=.001,
                 num_heads = 8, activation= "gelu", num_encoders = 4, num_classes=10):
        
        """hbye 未使用到 num_patches 和 patch——size  因为 我们bottleneck得到的输入 hw 极小了，所以不需要分patch"""
        super(VitBottleNeck, self).__init__()
        # self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches, dropout)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=3, stride=1, padding=1),
            nn.Flatten(2)
        )
        self.position_embed = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
        self.flatten = nn.Flatten(2)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, 1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 1)
        self.attn = nn.Conv2d(embed_dim, embed_dim, 5, padding=2, groups=embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.gelu = nn.GELU()
        # mlp_hidden_embed_dim = int(embed_dim * mlp_ratio)
        
        mlp_ratio = 4
        # self.mlp = CMlp(in_features=embed_dim, hidden_features=embed_dim * mlp_ratio)
        self.SelfAttn = SelfAttn(embed_dim, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.)
        
        ## 用不到  官方的encoder了
        # encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
        #                                            activation=activation,
        #                                            batch_first=True, norm_first=True)
        # self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
    
    def forward(self, x):
        # y = x
        B, N, H, W = x.shape
        # x = self.patch_embedding(x)
        ##### 仍是4d 
        # x = x + self.position_embed(x)
        # x_hidden = self.attn(self.conv1(self.norm1(x)))
        # x = x + self.drop_path(self.gelu(self.conv2(self.gelu(x_hidden))))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        #####
        #### 3d
        # x = x.flatten(2).transpose(1,2) #  0， 1，2 中 的1， 2 调换 B N C
        # x = self.encoder_blocks(x)
        x = self.SelfAttn(x)
        # print(x.shape)
        # # Assuming x shape is (BS, L, dim) where L = (z*z) + 1 and dim = c * w * w
        # BS, L, dim = x.shape  # 这里没有了cls token
        
        
        # # Extracting z and w from dim assuming dim = c * w * w
        # c = self.patch_embedding.patcher[0].out_channels
        # w = int(math.sqrt(dim // c))
        # z = int(math.sqrt(L))
        
        # # Reshaping x to (BS, c, z*w, z*w)
        # x = x.reshape(BS, z, z, c, w, w)
        # x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # Rearrange dimensions
        # x = x.view(BS, c, z * w, z * w)
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Vision Transformer (ViT) Training Script")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--train_df_dir', type=str, default="./dataset/train.csv", help='Path to train data')
    parser.add_argument('--test_df_dir', type=str, default="./dataset/test.csv", help='Path to test data')
    parser.add_argument('--submission_df_dir', type=str, default="./dataset/sample_submission.csv", help='Path to submission data')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--out_img_size', type=int, default=16, help='Input image size 256 /2/2/2/2 = 256/16')
    parser.add_argument('--patch_size', type=int, default=1, help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension') # in_channels * p1 * p2  cpp  = 1* 16 * 16
    parser.add_argument('--dropout', type=float, default=0.001, help='Dropout rate')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--activation', type=str, default="gelu", help='Activation function')
    parser.add_argument('--num_encoders', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd"], help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--datasetname', type=str, default="MNIST", help='Name of the dataset')
    parser.add_argument('--modelname', type=str, default="vit", help='Name of the model')
    parser.add_argument('--medmnist_name', type=str, default="breastmnist", help='Name of the medmnist data')
    
    return parser

        
# if __name__ == "__main__":# Parse command line arguments
#     parser = get_arg_parser()  # 由于是大通道 小hw 所以 我们还是 设置 patchsize 1
#     args = parser.parse_args()# Calculate num_patches
#     num_patches = (args.img_size // args.patch_size) ** 2# Initialize ViT model
#     embed_dim = args.in_channels * args.patch_size *args.patch_size
#     model = VitBottleNeck(args.in_channels, args.patch_size, embed_dim, num_patches, args.dropout,
#                           args.num_heads, args.activation, args.num_encoders, args.num_classes)# Print model parametersprint(f"Model Parameters: {count_params(model)}")# Create a random input tensor
#     x = torch.randn(size=(args.batch_size, args.in_channels, args.img_size, args.img_size))# Forward pass
#     output = model(x)# Print output shapeprint(f"Output shape: {output.shape}")
    

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


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMUNeXt_Vit_SelfAttn(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 256, 768], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7], args_vit = None):
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
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        if args_vit == None:
            PATCH_SIZE=1
            NUM_PATCHES = (16 //PATCH_SIZE) **2
        else:
            PATCH_SIZE = args_vit.patch_size
            NUM_PATCHES = (args_vit.out_img_size // PATCH_SIZE) ** 2
        IN_CHANNELS = dims[4]
        EMBED_DIM =dims[4]
        
        # Bottleneck
        # self.vit = VitBottleNeck(image_size=16, patch_size=4, num_classes=dims[4], dim=dims[4], depth=6, heads=8, mlp_dim=dims[4] * 4, channels=dims[4])
        # self.vit = VitBottleNeck(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT, NUM_HEADS, ACTIVATION, NUM_ENCODERS, NUM_CLASSES)
        self.vit = VitBottleNeck(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES) # EMBED_DIM = 768 ,
        
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

        # mlp (多次 conv norm)
        self.mlp_ratio = 4 ##hbye todo #TODO
        self.conv_bottleneck = CMlp(in_features=dims[4], hidden_features=self.mlp_ratio * dims[4], act_layer=nn.GELU, drop=0.)
        
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
        

        # Bottleneck
        x5 = self.vit(x5)
        # x5 = x5.view(x5.size(0), -1, 8, 8)  # Reshape to (BS, dims[4], 8, 8)
        # x5 = x5.view(x5.size(0), -1, 16, 16)  # Reshape to (BS, dims[4], 8, 8)
        # x5 = self.conv_bottleneck(x5)

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


import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class AgentAttentionBottleNeck(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        
        
    def forward(self, x):
        ### x 是 4d
        H = x.shape[2]
        W = x.shape[3]
        
        
# class CMUNeXt_Vit_SelfAttn(nn.Module):
#     def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 256, 768], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7], args_vit = None):
#         """
#         Args:
#             input_channel : input channel.
#             num_classes: output channel.
#             dims: length of channels
#             depths: length of cmunext blocks
#             kernels: kernal size of cmunext blocks
#         """
#         super().__init__()
#         # Encoder
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
#         self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
#         self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
#         self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
#         self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
#         self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
#         if args_vit == None:
#             PATCH_SIZE=1
#             NUM_PATCHES = (16 //PATCH_SIZE) **2
#         else:
#             PATCH_SIZE = args_vit.patch_size
#             NUM_PATCHES = (args_vit.out_img_size // PATCH_SIZE) ** 2
#         IN_CHANNELS = dims[4]
#         EMBED_DIM =dims[4]
        
#         # Bottleneck
#         # self.vit = VitBottleNeck(image_size=16, patch_size=4, num_classes=dims[4], dim=dims[4], depth=6, heads=8, mlp_dim=dims[4] * 4, channels=dims[4])
#         # self.vit = VitBottleNeck(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT, NUM_HEADS, ACTIVATION, NUM_ENCODERS, NUM_CLASSES)
#         self.vit = VitBottleNeck(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES) # EMBED_DIM = 768 ,
#         self.agent_attn = AgentAttentionBottleNeck(dim = dims[4], num_patches=16**2)
#         # Decoder
#         self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
#         self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
#         self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
#         self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
#         self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
#         self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
#         self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
#         self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
#         self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

#         # mlp (多次 conv norm)
#         self.mlp_ratio = 4 ##hbye todo #TODO
#         self.conv_bottleneck = CMlp(in_features=dims[4], hidden_features=self.mlp_ratio * dims[4], act_layer=nn.GELU, drop=0.)
        
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
        

#         # Bottleneck
#         x5 = self.agent_attn(x5)
#         # x5 = x5.view(x5.size(0), -1, 8, 8)  # Reshape to (BS, dims[4], 8, 8)
#         # x5 = x5.view(x5.size(0), -1, 16, 16)  # Reshape to (BS, dims[4], 8, 8)
#         # x5 = self.conv_bottleneck(x5)

#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.Up_conv5(d5)

#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)
#         d1 = self.Conv_1x1(d2)

#         return d1


# if __name__ == '__main__':
#     dim = 64
#     num_patches = 49

#     block = AgentAttention(dim=dim, num_patches=num_patches)

#     H, W = 7, 7
#     x = torch.rand(1, num_patches, dim)

#     # Forward pass
#     output = block(x, H, W)
#     print(f"Input size: {x.size()}")
#     print(f"Output size: {output.size()}")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# if __name__ == '__main__':
#     from loguru import logger as log
#     x = torch.randn(3, 3, 64, 64)  # 输入 B C H W
#     model = TinyUNet(in_channels=3, num_classes=1) # 这个就是 num——classes 的魅力
#     # print(model)
#     log.add('/data/hongboye/projects/model/plugs/segmentation/tinyUnet.log')
#     log.info("tinyUNet")
#     log.info(count_params(model))
#     log.info(model.named_modules)
#     output = model(x)
    
#     log.info("input shape", x.shape, "out.shape", output.shape)
    
#     print(output.shape)  # 输出应为 (3, 1, 64, 64)
# 示例使用
if __name__ == "__main__":
    parser = get_arg_parser()
    args_vit =  parser.parse_args()
    model = CMUNeXt(input_channel=3,args_vit=args_vit,num_classes=1 )
    x = torch.randn(2, 3, 256, 256)
    print(count_params(model))
    
    output = model(x)
    print(output.shape)