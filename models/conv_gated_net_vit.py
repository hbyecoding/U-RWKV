import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple


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

class DWConvWithGate(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[3, 3], expansion_factor=4):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor

        # Ensure that there are exactly two branches
        assert len(kernel_sizes) == 2, "There should be exactly two kernel sizes"

        # Branches with Depthwise Separable Convolutions and Bottleneck Structure
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2  # Maintain input size
            branch = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.GELU(),  # Use GELU instead of ReLU for consistency
                
                # Pointwise Convolution (expansion)
                nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1),
                nn.BatchNorm2d(in_channels * expansion_factor),
                nn.GELU(),
                
                # Pointwise Convolution (projection)
                nn.Conv2d(in_channels * expansion_factor, in_channels//2, kernel_size=1),
                nn.BatchNorm2d(in_channels //2),
                nn.GELU()
            )
            self.branches.append(branch)

        # Gating mechanism: 1x1 convolution to control information flow
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()  # Sigmoid activation for gating
        )

    def forward(self, x):
        # Apply multiple convolutional branches
        branch_outputs = [branch(x) for branch in self.branches]
        # Concatenate branch outputs along the channel dimension
        concat_output = torch.cat(branch_outputs, dim=1)

        # Apply gating mechanism
        gate_output = self.gate(concat_output)
        gated_output = concat_output * gate_output  # Element-wise multiplication

        # Sum the gated output to match the original channel size
        output = gated_output.sum(dim=1).unsqueeze(1)
        return output

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class GatedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, kernel_sizes=[3, 3], expansion_factor=4):
        super(GatedBlock, self).__init__()
        self.block = nn.Sequential(
            *[Residual(DWConvWithGate(ch_in, kernel_sizes, expansion_factor)) for _ in range(depth)],
            nn.Conv2d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )
        self.up = UpConv(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        # x = self.up(x)
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

class LoG_vit(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 5]]*5, expansion_factors=[2]*5,
                 args_vit = None):
        super(LoG_vit, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = ConvBlock(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = GatedBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], kernel_sizes=kernel_sizes[0], expansion_factor=expansion_factors[0])
        self.encoder2 = GatedBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], kernel_sizes=kernel_sizes[1], expansion_factor=expansion_factors[1])
        self.encoder3 = GatedBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], kernel_sizes=kernel_sizes[2], expansion_factor=expansion_factors[2])
        self.encoder4 = GatedBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], kernel_sizes=kernel_sizes[3], expansion_factor=expansion_factors[3])
        self.encoder5 = GatedBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], kernel_sizes=kernel_sizes[4], expansion_factor=expansion_factors[4])

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
        self.vit = VitBottleNeck(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES) 
        
        # Decoder
        self.Up5 = UpConv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = FusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = UpConv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = FusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = UpConv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = FusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = UpConv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = FusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
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

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Define dimensions and kernels
    dims = [48, 96, 192, 384, 768]
    kernels = [[3, 5]]*5

    
    # Create the U-Net model
    model = LoG_vit(kernel_sizes=kernels)
    
    # Print number of parameters
    print(f"Number of trainable parameters: {count_params(model)}")
    
    # Test the model with a sample input
    x = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image
    output = model(x)
    
    # Print the shape of the output
    print(f"Output shape: {output.shape}")
    
    print("END")